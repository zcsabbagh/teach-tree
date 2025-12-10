import os
import asyncio
import random
import logging
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from dotenv import load_dotenv
import anthropic

load_dotenv(".env.local")
logger = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-haiku-4-5"
MAX_RETRIES = 5
BASE_DELAY = 1.0
MAX_DELAY = 60.0


@dataclass
class CompletionResponse:
    content: str
    model: str
    usage: Dict[str, int]
    raw_response: Any


def _should_retry(error: Exception, attempt: int) -> bool:
    if attempt >= MAX_RETRIES:
        return False
    if isinstance(error, anthropic.RateLimitError):
        return True
    if isinstance(error, anthropic.APIConnectionError):
        return True
    if isinstance(error, anthropic.InternalServerError):
        return True
    return False


def _backoff(attempt: int) -> float:
    delay = min(BASE_DELAY * (2 ** attempt), MAX_DELAY)
    return delay * (0.5 + random.random())


def _parse_response(response: Any) -> CompletionResponse:
    content = ""
    for block in response.content:
        if block.type == "text":
            content += block.text
    return CompletionResponse(
        content=content,
        model=response.model,
        usage={
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
        },
        raw_response=response,
    )


def complete(
    prompt: str,
    *,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.7,
    max_tokens: int = 1024,
    system_prompt: Optional[str] = None,
) -> CompletionResponse:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not found in environment")

    client = anthropic.Anthropic(api_key=api_key)

    kwargs: Dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": [{"role": "user", "content": prompt}],
    }
    if system_prompt:
        kwargs["system"] = system_prompt

    for attempt in range(MAX_RETRIES + 1):
        try:
            response = client.messages.create(**kwargs)
            return _parse_response(response)
        except Exception as e:
            if not _should_retry(e, attempt):
                raise
            delay = _backoff(attempt)
            logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay:.2f}s...")
            import time
            time.sleep(delay)
    raise RuntimeError("Max retries exceeded")


async def acomplete(
    prompt: str,
    *,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.7,
    max_tokens: int = 1024,
    system_prompt: Optional[str] = None,
) -> CompletionResponse:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not found in environment")

    client = anthropic.AsyncAnthropic(api_key=api_key)

    kwargs: Dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": [{"role": "user", "content": prompt}],
    }
    if system_prompt:
        kwargs["system"] = system_prompt

    for attempt in range(MAX_RETRIES + 1):
        try:
            response = await client.messages.create(**kwargs)
            return _parse_response(response)
        except Exception as e:
            if not _should_retry(e, attempt):
                raise
            delay = _backoff(attempt)
            logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay:.2f}s...")
            await asyncio.sleep(delay)
    raise RuntimeError("Max retries exceeded")
