"""A transparent proxy for OpenAI's chat completions API endpoint."""

__version__ = "0.1.0"

from .config import load_config
from .api import app
from .utils import ThinkingTagFilter

from .backends import call_backend
from .streaming import stream_with_role
from .aggregation import aggregate_responses, progress_streaming_aggregator
from .utils import strip_thinking_tags
