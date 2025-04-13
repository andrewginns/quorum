"""
This module is maintained for backward compatibility.
It re-exports all components from the new modular structure.
"""

# Re-export all components from the new modular structure
from .config import load_config, config, OPENAI_API_BASE, DEFAULT_MODEL, TIMEOUT, aggregation_logger
from .api import app, proxy_chat_completions, health_check
from .streaming import StreamingResponseWrapper, stream_with_role
from .backends import call_backend
from .utils import strip_thinking_tags, ThinkingTagFilter
from .aggregation import aggregate_responses, progress_streaming_aggregator
from .models import Message, ChatCompletionRequest, ChatCompletionResponse, Choice, Delta

# Import standard libraries for backward compatibility
import httpx
import logging
import json
import yaml
import asyncio
import re
import os
from pathlib import Path
from typing import Dict, Any, AsyncGenerator, List
from fastapi import FastAPI, Request, Response
from fastapi.responses import StreamingResponse

# Create logger for backward compatibility
logger = logging.getLogger(__name__)

# Create async HTTP client for backward compatibility
http_client = httpx.AsyncClient()
