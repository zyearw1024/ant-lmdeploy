import logging
import re
import orjson
from contextvars import ContextVar
from typing import Optional, Sequence, Tuple, Union, List

from starlette.responses import StreamingResponse
from starlette.types import Send
from argparse import _SubParsersAction, ArgumentParser
from functools import cache
from lmdeploy.cli.serve import SubCliServe
from lmdeploy.serve.openai import api_server as openai_api_serve
from lmdeploy.serve.openai.protocol import (
    StreamOptions,
    DeltaMessage,
    ChatCompletionRequest,
)

logger = logging.getLogger(__name__)

# Context variable for request-level enable_thinking status
# This ensures proper isolation between concurrent requests
enable_thinking_context: ContextVar[Optional[bool]] = ContextVar(
    "enable_thinking_context", default=None
)

# Compile the regex for checking /think or /nothink at the end of the string
THINKING_TAG_REGEX = re.compile(r"/(think|nothink|no_think)\s*$")


class ApiServeCliContext:
    args = None


def _set_api_serve_cli_context(args):
    """Set the API serve CLI context with the provided arguments."""
    ApiServeCliContext.args = args


@cache
def get_stream_include_usage_status():
    """Get the status of whether to include stream usage data in the output."""
    try:
        args = ApiServeCliContext.args
        if args is None:
            return False
        return getattr(args, "enable_stream_include_usage", False)
    except:
        return False


@cache
def get_args_qwen3_enable_prompt_suffix_thinking():
    """Get the status of Qwen3 prompt suffix thinking mode (soft switch)."""
    try:
        args = ApiServeCliContext.args
        if args is None:
            return False
        # Check new parameter first, then fall back to old parameter for compatibility
        if hasattr(args, "qwen3_enable_prompt_suffix_thinking"):
            return getattr(args, "qwen3_enable_prompt_suffix_thinking", False)
        elif hasattr(args, "qwen3_enable_thinking"):
            return getattr(args, "qwen3_enable_thinking", False)
        return False
    except:
        return False


@cache
def get_args_qwen3_enable_chat_template_thinking():
    """Get the status of Qwen3 chat template thinking mode (hard switch)."""
    try:
        args = ApiServeCliContext.args
        if args is None:
            return False
        return getattr(args, "qwen3_enable_chat_template_thinking", False)
    except:
        return False


@cache
def get_remove_first_think_chunk_status():
    """Get the status of whether to remove the first chunk if it contains '<think>'."""
    try:
        args = ApiServeCliContext.args
        if args is None:
            return False
        status = getattr(args, "remove_first_think_chunk", False)
        print(f"remove_first_think_chunk: {status}")
        return status
    except:
        return False


_origin_parse_args = ArgumentParser.parse_args


def _patch_parse_args(self, args=None, namespace=None):
    """Patch the parse_args method to set the API serve CLI context if the command is 'serve'."""
    parser_args = _origin_parse_args(self, args=args, namespace=namespace)
    command = getattr(parser_args, "command", None)
    if command != "serve":
        return parser_args
    _set_api_serve_cli_context(parser_args)

    return parser_args


def get_api_serve_cli_context():
    """Get the API serve CLI context parser."""
    api_server_parser = SubCliServe.subparsers.choices["api_server"]
    return api_server_parser


_origin_api_serve_check_request = openai_api_serve.check_request


def set_enable_thinking_context(enable_thinking: Optional[bool]):
    """
    Set the enable_thinking status in the current request context.

    This function ensures proper isolation between concurrent requests by using ContextVar,
    which automatically maintains separate contexts for each async task/request.
    """
    enable_thinking_context.set(enable_thinking)
    logger.debug(f"Set enable_thinking context to: {enable_thinking}")


def get_enable_thinking_context() -> Optional[bool]:
    """
    Get the enable_thinking status from the current request context.

    Returns the value specific to the current async task/request context,
    ensuring no cross-contamination between concurrent requests.
    """
    return enable_thinking_context.get(None)


def _handle_qwen3_prompt_suffix_thinking(request, qwen3_enable_prompt_suffix_thinking):
    """Handle Qwen3 prompt suffix thinking logic (soft switch)."""
    try:
        # If qwen3 prompt suffix thinking is not enabled, return
        if not qwen3_enable_prompt_suffix_thinking:
            return

        # Ensure messages exist and is a list
        if not request.messages or not isinstance(request.messages, list):
            return

        last_message = request.messages[-1]

        # Ensure the last message is a dictionary and has content
        if not isinstance(last_message, dict) or "content" not in last_message:
            return

        content = last_message["content"]
        model_name = request.model
        _enable_think = True if model_name.endswith("-think") else False
        if _enable_think:
            # Strip -think suffix from model name
            request.model = model_name.rstrip("-think")
        # Check if content already ends with /think or /nothink followed by optional whitespace
        if THINKING_TAG_REGEX.search(content):
            return

        # Determine the tag to add based on model name
        if _enable_think:
            last_message["content"] += " /think"
        else:
            last_message["content"] += " /no_think"

    except Exception as e:
        logger.error(f"Error processing qwen3_prompt_suffix_thinking: {e}")


def _handle_qwen3_chat_template_thinking(request):
    """Handle Qwen3 chat template thinking logic (hard switch)."""
    try:
        # Initialize enable_thinking flag
        _enable_think = False

        # Handle model name compatibility: support -think suffix like soft switch
        if hasattr(request, "model") and request.model:
            model_name = request.model
            # Check if model name ends with '-think' (enable thinking)
            _enable_think = model_name.endswith("-think")
            if _enable_think:
                # Strip -think suffix from model name for compatibility
                request.model = model_name.rstrip("-think")

        # Hard switch: always override enable_thinking based on model name suffix
        # Only enable thinking if model name ends with '-think'
        original_value = getattr(request, "enable_thinking", None)
        request.enable_thinking = _enable_think
        logger.debug(
            f"Applied Qwen3 chat template thinking (hard switch): "
            f"enable_thinking={_enable_think} (was: {original_value})"
        )

    except Exception as e:
        logger.debug(f"Error processing qwen3 chat template thinking: {e}")


def handle_qwen3_thinking_modes(request):
    """
    Handle all Qwen3 thinking mode logic (both hard and soft switches).
    Enforces mutual exclusion between different thinking modes.
    Default behavior: no thinking mode enabled unless explicitly requested.
    """
    try:
        qwen3_chat_template_thinking = get_args_qwen3_enable_chat_template_thinking()
        qwen3_prompt_suffix_thinking = get_args_qwen3_enable_prompt_suffix_thinking()

        # Mutual exclusion: hard switch takes priority over soft switch
        if qwen3_chat_template_thinking and qwen3_prompt_suffix_thinking:
            logger.debug("Both Qwen3 thinking modes enabled, using hard switch only")
            qwen3_prompt_suffix_thinking = False

        # Process thinking modes in order of priority
        if qwen3_chat_template_thinking:
            _handle_qwen3_chat_template_thinking(request)
        elif qwen3_prompt_suffix_thinking:
            _handle_qwen3_prompt_suffix_thinking(request, qwen3_prompt_suffix_thinking)
        else:
            # Default behavior: no thinking mode enabled
            logger.debug("No Qwen3 thinking modes enabled, using default behavior")

    except Exception as e:
        logger.debug(f"Error handling qwen3 thinking modes: {e}")


_origin_stream_response = StreamingResponse.stream_response


# BUG: This is a temporary patch to handle the first chunk of the stream
# which might contain a "<think>" tag that needs to be ignored.
def _check_and_handle_first_chunk(chunk: bytes) -> bool:
    """
    Checks if the current chunk contains the "<think>" tag and should be ignored.

    Args:
        chunk: The current chunk of data.

    Returns:
        A boolean indicating if the current chunk should be ignored.
    """
    ignore_first_chunk = False
    try:
        # Attempt to parse the chunk as JSON
        parser_chunk = chunk.lstrip(b"data:")
        chunk_data = orjson.loads(parser_chunk)
        choices = chunk_data.get("choices", [])
        if choices:
            delta = choices[0].get("delta", {})
            content = delta.get("content")
            if content == "<think>":
                # logger.debug("Ignoring first chunk containing '<think>'")
                ignore_first_chunk = True
    except orjson.JSONDecodeError:
        # Handle cases where the first chunk is not valid JSON
        logger.warning(
            "Failed to decode first chunk as JSON. Proceeding without ignoring."
        )
    except Exception as e:
        # Catch any other unexpected errors during processing
        logger.error(
            f"An unexpected error occurred while processing the first chunk: {e}"
        )

    return ignore_first_chunk


async def _patch_stream_response(self, send: Send) -> None:
    await send(
        {
            "type": "http.response.start",
            "status": self.status_code,
            "headers": self.raw_headers,
        }
    )

    _is_first_chunk = True
    async for chunk in self.body_iterator:
        if not isinstance(chunk, (bytes, memoryview)):
            chunk = chunk.encode(self.charset)

        ignore_first_chunk = False
        if _is_first_chunk and get_remove_first_think_chunk_status():
            ignore_first_chunk = _check_and_handle_first_chunk(chunk)
        _is_first_chunk = False

        if not ignore_first_chunk:
            await send({"type": "http.response.body", "body": chunk, "more_body": True})

    await send({"type": "http.response.body", "body": b"", "more_body": False})


def _patch_api_serve_check_request(request):
    """
    Patch the check_request method to include stream usage data if enabled.

    This function is called at the beginning of each request and sets up the
    request-specific context using ContextVar for proper isolation.
    """
    logger.info("Entering _patch_api_serve_check_request")
    enable_stream_include_usage_status = get_stream_include_usage_status()
    if enable_stream_include_usage_status:
        stream_options = getattr(request, "stream_options", None)
        if not stream_options:
            stream_options = StreamOptions(include_usage=True)
            request.stream_options = stream_options
            logger.info("Enabled stream_options.include_usage")

    # logger.info(f"Request before handling qwen3 thinking modes: {request.json()}")
    handle_qwen3_thinking_modes(request)
    # logger.info(f"Request after handling qwen3 thinking modes: {request.json()}")

    # Set enable_thinking context based on CLI args and request
    # This context will be isolated per request/async task
    enable_thinking_value = None

    # Check if any thinking modes are enabled or user manually passed the value
    qwen3_chat_template_thinking = get_args_qwen3_enable_chat_template_thinking()
    qwen3_prompt_suffix_thinking = get_args_qwen3_enable_prompt_suffix_thinking()
    user_enable_thinking = getattr(request, "enable_thinking", None)

    if (
        qwen3_chat_template_thinking
        or qwen3_prompt_suffix_thinking
        or user_enable_thinking is not None
    ):
        # Priority: user manual input > chat template thinking > prompt suffix thinking
        if user_enable_thinking is not None:
            enable_thinking_value = user_enable_thinking
            logger.debug(
                f"Using user-provided enable_thinking: {enable_thinking_value}"
            )
        elif qwen3_chat_template_thinking:
            # For chat template thinking, get the value from the request after processing
            enable_thinking_value = getattr(request, "enable_thinking", None)
            logger.debug(
                f"Using chat template thinking enable_thinking: {enable_thinking_value}"
            )
        elif qwen3_prompt_suffix_thinking:
            # For prompt suffix thinking, we assume thinking is enabled if any thinking mode is on
            enable_thinking_value = True
            logger.debug(
                f"Using prompt suffix thinking enable_thinking: {enable_thinking_value}"
            )

    # Set the context - this will be isolated per async task/request
    set_enable_thinking_context(enable_thinking_value)
    logger.debug(f"Final enable_thinking context set to: {enable_thinking_value}")

    r = _origin_api_serve_check_request(request)
    logger.debug("Exiting _patch_api_serve_check_request")
    return r


def _patch_api_server_add_parser(parser):
    """Patch the API server parser to add necessary arguments."""
    parser.add_argument(
        "--enable-stream-include-usage",
        action="store_true",
        help="Enable the inclusion of stream usage data in the output, useful for monitoring performance.",
    )

    # Backward compatibility: keep old parameter name
    parser.add_argument(
        "--qwen3-enable-thinking",
        action="store_true",
        default=False,
        help="[DEPRECATED] Use --qwen3-enable-prompt-suffix-thinking instead. "
        "Enable Qwen3 thinking mode. If not set, will check environment variable QWEN3_ENABLE_THINKING. Default: off. "
        "If model name is Qwen3-1.7B, thinking is off by default. If model name is Qwen3-1.7B-think, thinking is on.",
    )

    parser.add_argument(
        "--qwen3-enable-prompt-suffix-thinking",
        action="store_true",
        default=False,
        help="Enable Qwen3 thinking mode by appending `/think` or `/nothink` suffix to the prompt (soft switch). "
        "When enabled, models ending with '-think' will automatically append '/think', "
        "while other models will append '/no_think'. Default: off.",
    )

    parser.add_argument(
        "--qwen3-enable-chat-template-thinking",
        action="store_true",
        default=False,
        help="Enable Qwen3 chat template thinking mode (hard switch). "
        "When enabled, it will set request.enable_thinking "
        "if not specified in the request. "
        "This provides stronger constraints than the soft switch. "
        "Mutually exclusive with --qwen3-enable-prompt-suffix-thinking. Default: off.",
    )

    parser.add_argument(
        "--remove-first-think-chunk",
        action="store_true",
        default=False,
        help="Enable removal of the first chunk if it contains the '<think>' tag.",
    )

    return parser


def _patch_sub_parser_add_parser(self, name, **kwargs):
    """Patch the subparser add_parser method to include the --enable-stream-include-usage argument for 'api_server'."""
    _parser = self._origin_sub_parser_add_parser(name, **kwargs)

    if name != "api_server":
        return _parser
    logger.info("patching api_server add_parser")
    _parser = _patch_api_server_add_parser(_parser)

    return _parser


def _patched_extract_reasoning_content_streaming(
    self,
    previous_text: str,
    current_text: str,
    delta_text: str,
    previous_token_ids: Sequence[int],
    current_token_ids: Sequence[int],
    delta_token_ids: Sequence[int],
    **kwargs,
) -> Union[DeltaMessage, None]:
    """
    Patched version that checks enable_thinking context.

    This function respects the request-specific context set by ContextVar,
    ensuring proper isolation between concurrent requests.
    """
    # Check enable_thinking context - this is isolated per request
    enable_thinking = get_enable_thinking_context()

    # If enable_thinking is explicitly False, implement special handling
    if enable_thinking is False:
        return DeltaMessage(content=delta_text)

    # Call original method for normal cases
    return self._origin_extract_reasoning_content_streaming(
        previous_text,
        current_text,
        delta_text,
        previous_token_ids,
        current_token_ids,
        delta_token_ids,
        **kwargs,
    )


def _patched_extract_reasoning_content(
    self, model_output: str, request: ChatCompletionRequest, **kwargs
) -> Tuple[Optional[str], Optional[str]]:
    """
    Patched version that checks enable_thinking context.

    This function respects the request-specific context set by ContextVar,
    ensuring proper isolation between concurrent requests.
    """
    # Check enable_thinking context - this is isolated per request
    enable_thinking = get_enable_thinking_context()

    # If enable_thinking is explicitly False, return content only without reasoning
    if enable_thinking is False:
        return None, model_output

    # Call original method for normal cases
    return self._origin_extract_reasoning_content(model_output, request, **kwargs)


def _patch_deepseek_r1_reasoning_parser():
    """Patch DeepSeekR1ReasoningParser to respect enable_thinking context."""
    try:
        from lmdeploy.serve.openai.reasoning_parser.deepseek_r1_reasoning_parser import (
            DeepSeekR1ReasoningParser,
        )

        # Store original methods
        if not hasattr(
            DeepSeekR1ReasoningParser, "_origin_extract_reasoning_content_streaming"
        ):
            setattr(
                DeepSeekR1ReasoningParser,
                "_origin_extract_reasoning_content_streaming",
                DeepSeekR1ReasoningParser.extract_reasoning_content_streaming,
            )
        if not hasattr(DeepSeekR1ReasoningParser, "_origin_extract_reasoning_content"):
            setattr(
                DeepSeekR1ReasoningParser,
                "_origin_extract_reasoning_content",
                DeepSeekR1ReasoningParser.extract_reasoning_content,
            )

        # Apply patches
        DeepSeekR1ReasoningParser.extract_reasoning_content_streaming = (
            _patched_extract_reasoning_content_streaming
        )
        DeepSeekR1ReasoningParser.extract_reasoning_content = (
            _patched_extract_reasoning_content
        )

        logger.info("Successfully patched DeepSeekR1ReasoningParser methods")

    except ImportError as e:
        logger.warning(f"Could not import DeepSeekR1ReasoningParser for patching: {e}")
    except Exception as e:
        logger.error(f"Error patching DeepSeekR1ReasoningParser: {e}")


def patch_all():
    """Apply all monkey patches."""
    logger.info("monkey patching all")

    # Patch ArgumentParser - use try/except to handle type checker warnings
    try:
        if not hasattr(_SubParsersAction, "_origin_sub_parser_add_parser"):
            setattr(
                _SubParsersAction,
                "_origin_sub_parser_add_parser",
                _SubParsersAction.add_parser,
            )
        _SubParsersAction.add_parser = _patch_sub_parser_add_parser
    except Exception as e:
        logger.warning(f"Failed to patch _SubParsersAction: {e}")

    # Patch openai_api_serve
    openai_api_serve.check_request = _patch_api_serve_check_request

    # Patch ArgumentParser
    ArgumentParser.parse_args = _patch_parse_args

    # Always apply the patch to remove the first chunk, the behavior is controlled by get_remove_first_think_chunk_status() at runtime
    try:
        if not hasattr(StreamingResponse, "_origin_stream_response"):
            setattr(
                StreamingResponse, "_origin_stream_response", _origin_stream_response
            )
        StreamingResponse.stream_response = _patch_stream_response
        logger.info("Patch for removing the first chunk applied.")
    except Exception as e:
        logger.warning(f"Failed to patch StreamingResponse: {e}")

    # Patch DeepSeekR1ReasoningParser to respect enable_thinking context
    _patch_deepseek_r1_reasoning_parser()
