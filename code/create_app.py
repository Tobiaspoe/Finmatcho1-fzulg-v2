"""
This module creates a Flask app that serves the web interface for the chatbot.
"""

import functools
import json
import logging
import mimetypes
import sys
from os import path
import requests
from flask import Flask, Response, request, Request, jsonify
from openai import AzureOpenAI, Stream, APIStatusError
from openai.types.chat import ChatCompletionChunk
from dotenv import load_dotenv

# If you're keeping some helper logic (like env loading or orchestrator), keep these:
from backend.batch.utilities.helpers.env_helper import EnvHelper
from backend.batch.utilities.helpers.orchestrator_helper import Orchestrator
from backend.batch.utilities.helpers.config.config_helper import ConfigHelper
from backend.batch.utilities.helpers.config.conversation_flow import ConversationFlow
from backend.api.chat_history import bp_chat_history_response

ERROR_429_MESSAGE = "We're currently experiencing a high number of requests for the service you're trying to access. Please wait a moment and try again."
ERROR_GENERIC_MESSAGE = "An error occurred. Please try again. If the problem persists, please contact the site administrator."
logger = logging.getLogger(__name__)


def stream_response(response: Stream[ChatCompletionChunk]):
    """Stream OpenAI chat completion response without custom data/citations."""
    response_obj = {
        "id": "",
        "model": "",
        "created": 0,
        "object": "",
        "choices": [
            {
                "messages": [
                    {
                        "content": "",
                        "end_turn": False,
                        "role": "assistant",
                    }
                ]
            }
        ],
    }

    for line in response:
        choice = line.choices[0]

        if choice.model_extra.get("end_turn"):
            response_obj["choices"][0]["messages"][0]["end_turn"] = True
            yield json.dumps(response_obj, ensure_ascii=False) + "\n"
            return

        response_obj["id"] = line.id
        response_obj["model"] = line.model
        response_obj["created"] = line.created
        response_obj["object"] = line.object

        delta = choice.delta
        response_obj["choices"][0]["messages"][0]["content"] += delta.content or ""

        yield json.dumps(response_obj, ensure_ascii=False) + "\n"


    for line in response:
        choice = line.choices[0]

        if choice.model_extra["end_turn"]:
            response_obj["choices"][0]["messages"][1]["end_turn"] = True
            yield json.dumps(response_obj, ensure_ascii=False) + "\n"
            return

        response_obj["id"] = line.id
        response_obj["model"] = line.model
        response_obj["created"] = line.created
        response_obj["object"] = line.object

        delta = choice.delta
        role = delta.role

        if role == "assistant":
            citations = get_citations(delta.model_extra["context"])
            response_obj["choices"][0]["messages"][0]["content"] = json.dumps(
                citations,
                ensure_ascii=False,
            )
        else:
            response_obj["choices"][0]["messages"][1]["content"] += delta.content

        yield json.dumps(response_obj, ensure_ascii=False) + "\n"


def conversation_openai(conversation: Request, env_helper: EnvHelper):
    """This function streams the response from Azure OpenAI without using Azure Search."""
    logger.info("Method conversation_openai started")

    if env_helper.is_auth_type_keys():
        logger.info("Using key-based authentication for Azure OpenAI")
        openai_client = AzureOpenAI(
            azure_endpoint=env_helper.AZURE_OPENAI_ENDPOINT,
            api_version=env_helper.AZURE_OPENAI_API_VERSION,
            api_key=env_helper.AZURE_OPENAI_API_KEY,
        )
    else:
        logger.info("Using RBAC authentication for Azure OpenAI")
        openai_client = AzureOpenAI(
            azure_endpoint=env_helper.AZURE_OPENAI_ENDPOINT,
            api_version=env_helper.AZURE_OPENAI_API_VERSION,
            azure_ad_token_provider=env_helper.AZURE_TOKEN_PROVIDER,
        )

    request_messages = conversation.json["messages"]
    messages = []

    config = ConfigHelper.get_active_config_or_default()
    if config.prompts.use_on_your_data_format:
        messages.append(
            {"role": "system", "content": config.prompts.answering_system_prompt}
        )

    for message in request_messages:
        messages.append({"role": message["role"], "content": message["content"]})

    # Create the completion request WITHOUT Azure Search
    response = openai_client.chat.completions.create(
        model=env_helper.AZURE_OPENAI_MODEL,
        messages=messages,
        temperature=float(env_helper.AZURE_OPENAI_TEMPERATURE),
        max_tokens=int(env_helper.AZURE_OPENAI_MAX_TOKENS),
        top_p=float(env_helper.AZURE_OPENAI_TOP_P),
        stop=(
            env_helper.AZURE_OPENAI_STOP_SEQUENCE.split("|")
            if env_helper.AZURE_OPENAI_STOP_SEQUENCE
            else None
        ),
        stream=env_helper.SHOULD_STREAM,
    )

    if not env_helper.SHOULD_STREAM:
        response_obj = {
            "id": response.id,
            "model": response.model,
            "created": response.created,
            "object": response.object,
            "choices": [
                {
                    "messages": [
                        {
                            "end_turn": True,
                            "content": response.choices[0].message.content,
                            "role": "assistant",
                        }
                    ]
                }
            ],
        }
        return response_obj

    logger.info("Method conversation_openai ended")
    return Response(stream_response(response), mimetype="application/json-lines")

def stream_without_data(response: Stream[ChatCompletionChunk]):
    """This function streams the response from Azure OpenAI without data."""
    response_text = ""
    for line in response:
        if not line.choices:
            continue

        delta_text = line.choices[0].delta.content

        if delta_text is None:
            return

        response_text += delta_text

        response_obj = {
            "id": line.id,
            "model": line.model,
            "created": line.created,
            "object": line.object,
            "choices": [
                {"messages": [{"role": "assistant", "content": response_text}]}
            ],
        }
        yield json.dumps(response_obj, ensure_ascii=False) + "\n"


def get_message_orchestrator():
    """This function gets the message orchestrator."""
    return Orchestrator()


def get_orchestrator_config():
    """This function gets the orchestrator configuration."""
    return ConfigHelper.get_active_config_or_default().orchestrator


def conversation_without_data(conversation: Request, env_helper: EnvHelper):
    """This function streams the response from Azure OpenAI without data."""
    if env_helper.AZURE_AUTH_TYPE == "rbac":
        openai_client = AzureOpenAI(
            azure_endpoint=env_helper.AZURE_OPENAI_ENDPOINT,
            api_version=env_helper.AZURE_OPENAI_API_VERSION,
            azure_ad_token_provider=env_helper.AZURE_TOKEN_PROVIDER,
        )
    else:
        openai_client = AzureOpenAI(
            azure_endpoint=env_helper.AZURE_OPENAI_ENDPOINT,
            api_version=env_helper.AZURE_OPENAI_API_VERSION,
            api_key=env_helper.AZURE_OPENAI_API_KEY,
        )

    request_messages = conversation.json["messages"]
    messages = [{"role": "system", "content": env_helper.AZURE_OPENAI_SYSTEM_MESSAGE}]

    for message in request_messages:
        messages.append({"role": message["role"], "content": message["content"]})

    # Azure Open AI takes the deployment name as the model name, "AZURE_OPENAI_MODEL" means
    # deployment name.
    response = openai_client.chat.completions.create(
        model=env_helper.AZURE_OPENAI_MODEL,
        messages=messages,
        temperature=float(env_helper.AZURE_OPENAI_TEMPERATURE),
        max_tokens=int(env_helper.AZURE_OPENAI_MAX_TOKENS),
        top_p=float(env_helper.AZURE_OPENAI_TOP_P),
        stop=(
            env_helper.AZURE_OPENAI_STOP_SEQUENCE.split("|")
            if env_helper.AZURE_OPENAI_STOP_SEQUENCE
            else None
        ),
        stream=env_helper.SHOULD_STREAM,
    )

    if not env_helper.SHOULD_STREAM:
        response_obj = {
            "id": response.id,
            "model": response.model,
            "created": response.created,
            "object": response.object,
            "choices": [
                {
                    "messages": [
                        {
                            "role": "assistant",
                            "content": response.choices[0].message.content,
                        }
                    ]
                }
            ],
        }
        return jsonify(response_obj), 200

    return Response(stream_without_data(response), mimetype="application/json-lines")


@functools.cache
def get_speech_key(env_helper: EnvHelper):
    """
    Get the Azure Speech key directly from Azure.
    This is required to generate short-lived tokens when using RBAC.
    """
    client = CognitiveServicesManagementClient(
        credential=DefaultAzureCredential(),
        subscription_id=env_helper.AZURE_SUBSCRIPTION_ID,
    )
    keys = client.accounts.list_keys(
        resource_group_name=env_helper.AZURE_RESOURCE_GROUP,
        account_name=env_helper.AZURE_SPEECH_SERVICE_NAME,
    )

    return keys.key1


def create_app():
    """This function creates the Flask app."""
    # Fixing MIME types for static files under Windows
    mimetypes.add_type("application/javascript", ".js")
    mimetypes.add_type("text/css", ".css")

    sys.path.append(path.join(path.dirname(__file__), ".."))

    load_dotenv(
        path.join(path.dirname(__file__), "..", "..", ".env")
    )  # Load environment variables from .env file

    app = Flask(__name__)
    env_helper: EnvHelper = EnvHelper()

    logger.debug("Starting web app")

    @app.route("/", defaults={"path": "index.html"})
    @app.route("/<path:path>")
    def static_file(path):
        return app.send_static_file(path)

    @app.route("/api/health", methods=["GET"])
    def health():
        return "OK"

    def conversation_azure_byod():
        logger.info("Method conversation_azure_byod started")
    try:
        return conversation_without_data(request, env_helper)
    except APIStatusError as e:
        error_message = str(e)
        logger.exception("Exception in /api/conversation | %s", error_message)
        response_json = e.response.json()
        response_message = response_json.get("error", {}).get("message", "")
        response_code = response_json.get("error", {}).get("code", "")
        if response_code == "429" or "429" in response_message:
            return jsonify({"error": ERROR_429_MESSAGE}), 429
        return jsonify({"error": ERROR_GENERIC_MESSAGE}), 500
    except Exception as e:
        error_message = str(e)
        logger.exception("Exception in /api/conversation | %s", error_message)
        return jsonify({"error": ERROR_GENERIC_MESSAGE}), 500
    finally:
        logger.info("Method conversation_azure_byod ended")

    async def conversation_custom():
        message_orchestrator = get_message_orchestrator()

        try:
            logger.info("Method conversation_custom started")
            user_message = request.json["messages"][-1]["content"]
            conversation_id = request.json["conversation_id"]
            user_assistant_messages = list(
                filter(
                    lambda x: x["role"] in ("user", "assistant"),
                    request.json["messages"][0:-1],
                )
            )

            messages = await message_orchestrator.handle_message(
                user_message=user_message,
                chat_history=user_assistant_messages,
                conversation_id=conversation_id,
                orchestrator=get_orchestrator_config(),
            )

            response_obj = {
                "id": "response.id",
                "model": env_helper.AZURE_OPENAI_MODEL,
                "created": "response.created",
                "object": "response.object",
                "choices": [{"messages": messages}],
            }

            return jsonify(response_obj), 200

        except APIStatusError as e:
            error_message = str(e)
            logger.exception("Exception in /api/conversation | %s", error_message)
            response_json = e.response.json()
            response_message = response_json.get("error", {}).get("message", "")
            response_code = response_json.get("error", {}).get("code", "")
            if response_code == "429" or "429" in response_message:
                return jsonify({"error": ERROR_429_MESSAGE}), 429
            return jsonify({"error": ERROR_GENERIC_MESSAGE}), 500
        except Exception as e:
            error_message = str(e)
            logger.exception("Exception in /api/conversation | %s", error_message)
            return jsonify({"error": ERROR_GENERIC_MESSAGE}), 500
        finally:
            logger.info("Method conversation_custom ended")

    @app.route("/api/conversation", methods=["POST"])
    async def conversation():
        ConfigHelper.get_active_config_or_default.cache_clear()
        result = ConfigHelper.get_active_config_or_default()
        conversation_flow = result.prompts.conversational_flow
        if conversation_flow == ConversationFlow.CUSTOM.value:
            return await conversation_custom()
        elif conversation_flow == ConversationFlow.BYOD.value:
            return conversation_azure_byod()
        else:
            return (
                jsonify(
                    {
                        "error": "Invalid conversation flow configured. Value can only be 'custom' or 'byod'."
                    }
                ),
                500,
            )

    @app.route("/api/speech", methods=["GET"])
    def speech_config():
        """Get the speech config for Azure Speech."""
        try:
            logger.info("Method speech_config started")
            speech_key = env_helper.AZURE_SPEECH_KEY or get_speech_key(env_helper)

            response = requests.post(
                f"{env_helper.AZURE_SPEECH_REGION_ENDPOINT}sts/v1.0/issueToken",
                headers={
                    "Ocp-Apim-Subscription-Key": speech_key,
                },
                timeout=5,
            )

            if response.status_code == 200:
                return {
                    "token": response.text,
                    "key": speech_key,
                    "region": env_helper.AZURE_SPEECH_SERVICE_REGION,
                    "languages": env_helper.AZURE_SPEECH_RECOGNIZER_LANGUAGES,
                }

            logger.error("Failed to get speech config: %s", response.text)
            return {"error": "Failed to get speech config"}, response.status_code
        except Exception as e:
            logger.exception("Exception in /api/speech | %s", str(e))

            return {"error": "Failed to get speech config"}, 500
        finally:
            logger.info("Method speech_config ended")

    @app.route("/api/assistanttype", methods=["GET"])
    def assistanttype():
        ConfigHelper.get_active_config_or_default.cache_clear()
        result = ConfigHelper.get_active_config_or_default()
        return jsonify({"ai_assistant_type": result.prompts.ai_assistant_type})

    @app.route("/api/checkauth", methods=["GET"])
    async def check_auth_enforced():
        """Check if the authentiction is enforced."""
        return jsonify({"is_auth_enforced": env_helper.ENFORCE_AUTH})

    app.register_blueprint(bp_chat_history_response, url_prefix="/api")
    return app
