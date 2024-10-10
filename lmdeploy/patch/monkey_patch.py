import logging

from argparse import _SubParsersAction, ArgumentParser
from functools import cache
from lmdeploy.cli.serve import SubCliServe
from lmdeploy.serve.openai import api_server as openai_api_serve
from lmdeploy.serve.openai.protocol import StreamOptions

logger = logging.getLogger(__name__)

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
def _patch_api_serve_check_request(request):
    """Patch the check_request method to include stream usage data if enabled."""
    enable_stream_include_usage_status = get_stream_include_usage_status()
    if enable_stream_include_usage_status:
        stream_options = getattr(request, "stream_options", None)
        if not stream_options:
            stream_options = StreamOptions(include_usage=True)
            request.stream_options = stream_options
            
    r = _origin_api_serve_check_request(request)
    return r
     
    
def _patch_api_server_add_parser(parser):
    """Patch the API server parser to add the --enable-stream-include-usage argument."""
    parser.add_argument(
        "--enable-stream-include-usage",
        action="store_true",
        help="Enable the inclusion of stream usage data in the output, useful for monitoring performance.",
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
