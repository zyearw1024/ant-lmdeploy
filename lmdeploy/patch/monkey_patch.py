import logging
import re
import orjson

from starlette.responses import StreamingResponse
from starlette.types import Send
from argparse import _SubParsersAction, ArgumentParser
from functools import cache
from lmdeploy.cli.serve import SubCliServe
from lmdeploy.serve.openai import api_server as openai_api_serve
from lmdeploy.serve.openai.protocol import StreamOptions

logger = logging.getLogger(__name__)

# Compile the regex for checking /think or /nothink at the end of the string
THINKING_TAG_REGEX = re.compile(r"/(think|nothink||no_think)\s*$")

class ApiServeCliContext:
    args = None
    
def _set_api_serve_cli_context(args):
    """Set the API serve CLI context with the provided arguments."""
    ApiServeCliContext.args = args
    
@cache
def get_stream_include_usage_status():
    """Get the status of whether to include stream usage data in the output."""
    try:
        return ApiServeCliContext.args.enable_stream_include_usage
    except:
        return False
    
@cache
def get_args_qwen3_enable_thinking():
    """Get the status of Qwen3 enable thinking mode."""
    try:
        return ApiServeCliContext.args.qwen3_enable_thinking
    except:
        return False

@cache
def get_remove_first_think_chunk_status():
    """Get the status of whether to remove the first chunk if it contains '<think>'."""
    try:
        status = getattr(ApiServeCliContext.args, "remove_first_think_chunk")
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
    api_server_parser = SubCliServe.subparsers.choices['api_server']
    return api_server_parser
    
    
_origin_api_serve_check_request = openai_api_serve.check_request


def _handle_qwen3_thinking(request, qwen3_enable_thinking):
    """Handle Qwen3 thinking logic based on model name and enable flag."""
    try:
        # If qwen3 thinking is not enabled, return
        if not qwen3_enable_thinking:
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
            #   # Patch SubCliServe
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
        logger.error(f"Error processing qwen3_enable_thinking: {e}")

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
        parser_chunk = chunk.lstrip(b'data:')
        chunk_data = orjson.loads(parser_chunk)
        choices = chunk_data.get("choices", [])
        if choices:
            delta = choices[0].get('delta', {})
            content = delta.get('content')
            if content == "<think>":
                # logger.debug("Ignoring first chunk containing '<think>'")
                ignore_first_chunk = True
    except orjson.JSONDecodeError:
        # Handle cases where the first chunk is not valid JSON
        logger.warning("Failed to decode first chunk as JSON. Proceeding without ignoring.")
    except Exception as e:
        # Catch any other unexpected errors during processing
        logger.error(f"An unexpected error occurred while processing the first chunk: {e}")

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
    """Patch the check_request method to include stream usage data if enabled."""
    logger.info("Entering _patch_api_serve_check_request")
    enable_stream_include_usage_status = get_stream_include_usage_status()
    if enable_stream_include_usage_status:
        stream_options = getattr(request, "stream_options", None)
        if not stream_options:
            stream_options = StreamOptions(include_usage=True)
            request.stream_options = stream_options
            logger.info("Enabled stream_options.include_usage")
    qwen3_enable_thinking = get_args_qwen3_enable_thinking()
    logger.info(f"qwen3_enable_thinking: {qwen3_enable_thinking}")
    logger.info(f"Request before handling qwen3 thinking: {request.json()}")
    _handle_qwen3_thinking(request, qwen3_enable_thinking)
    logger.info(f"Request after handling qwen3 thinking: {request.json()}")
    r = _origin_api_serve_check_request(request)
    logger.info("Exiting _patch_api_serve_check_request")
    return r
     
def _patch_api_server_add_parser(parser):
    """Patch the API server parser to add necessary arguments."""
    parser.add_argument(
        "--enable-stream-include-usage",
        action="store_true",
        help="Enable the inclusion of stream usage data in the output, useful for monitoring performance.",
    )
    parser.add_argument(
        "--qwen3-enable-thinking",
        action="store_true",
        default=False,
        help="Enable Qwen3 thinking mode. If not set, will check environment variable QWEN3_ENABLE_THINKING. Default: off. "
             "If model name is Qwen3-1.7B, thinking is off by default. If model name is Qwen3-1.7B-think, thinking is on."
    )
    parser.add_argument(
        "--remove-first-think-chunk",
        action="store_true",
        default=False,
        help="Enable removal of the first chunk if it contains the '<think>' tag."
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

def patch_all():
    """Apply all monkey patches."""
    logger.info("monkey patching all")

    # Patch ArgumentParser
    _SubParsersAction._origin_sub_parser_add_parser = _SubParsersAction.add_parser
    _SubParsersAction.add_parser = _patch_sub_parser_add_parser
 
    # Patch openai_api_serve
    openai_api_serve.check_request = _patch_api_serve_check_request
    
    # Patch ArgumentParser
    ArgumentParser.parse_args = _patch_parse_args

    # Always apply the patch to remove the first chunk, the behavior is controlled by get_remove_first_think_chunk_status() at runtime
    StreamingResponse._origin_stream_response = _origin_stream_response
    StreamingResponse.stream_response = _patch_stream_response
    logger.info("Patch for removing the first chunk applied.")