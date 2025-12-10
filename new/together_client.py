import os
import asyncio
import random
import time
import logging
from collections import deque
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from dotenv import load_dotenv
from together import Together, AsyncTogether
from together.error import RateLimitError, APIConnectionError, ServiceUnavailableError, APIError

load_dotenv(".env.local")
logger = logging.getLogger(__name__)

DEFAULT_MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"
MAX_RETRIES = 5
BASE_DELAY = 1.0
MAX_DELAY = 60.0

# Rate limiting: Together AI allows 10 QPS
MAX_QPS = 8  # Stay under 10 for safety margin
_rate_limiter: Optional["RateLimiter"] = None


class RateLimiter:
    """Token bucket rate limiter for API calls."""

    def __init__(self, max_qps: float = MAX_QPS):
        self.max_qps = max_qps
        self.min_interval = 1.0 / max_qps
        self._timestamps: deque = deque()
        self._lock = asyncio.Lock()

    async def acquire(self):
        """Wait until we can make a request without exceeding rate limit."""
        async with self._lock:
            now = time.monotonic()

            # Remove timestamps older than 1 second
            while self._timestamps and now - self._timestamps[0] > 1.0:
                self._timestamps.popleft()

            # If we're at the limit, wait
            if len(self._timestamps) >= self.max_qps:
                wait_time = 1.0 - (now - self._timestamps[0]) + 0.01  # small buffer
                if wait_time > 0:
                    logger.debug(f"Rate limit: waiting {wait_time:.2f}s ({len(self._timestamps)} requests in last second)")
                    await asyncio.sleep(wait_time)
                    now = time.monotonic()
                    # Clean up again after waiting
                    while self._timestamps and now - self._timestamps[0] > 1.0:
                        self._timestamps.popleft()

            self._timestamps.append(now)


def get_rate_limiter() -> RateLimiter:
    """Get or create the global rate limiter."""
    global _rate_limiter
    if _rate_limiter is None:
        _rate_limiter = RateLimiter()
    return _rate_limiter


@dataclass
class CompletionResponse:
    content: str
    model: str
    usage: Dict[str, int]
    finish_reason: str
    raw_response: Any


def _should_retry(error: Exception, attempt: int) -> bool:
    if attempt >= MAX_RETRIES:
        return False
    if isinstance(error, RateLimitError):
        return True
    if isinstance(error, APIConnectionError):
        return True
    if isinstance(error, ServiceUnavailableError):
        return True
    # Retry on APIError (includes 500/502/503 from Cloudflare)
    if isinstance(error, APIError):
        return True
    return False


def _backoff(attempt: int) -> float:
    delay = min(BASE_DELAY * (2 ** attempt), MAX_DELAY)
    return delay * (0.5 + random.random())


def _build_messages(prompt: str, system_prompt: Optional[str] = None) -> List[Dict[str, str]]:
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    return messages


def _parse_response(response: Any) -> CompletionResponse:
    return CompletionResponse(
        content=response.choices[0].message.content,
        model=response.model,
        usage={
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        },
        finish_reason=response.choices[0].finish_reason,
        raw_response=response,
    )


def complete(
    prompt: str,
    *,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
    system_prompt: Optional[str] = None,
) -> CompletionResponse:
    api_key = os.getenv("TOGETHER_API_KEY")
    if not api_key:
        raise ValueError("TOGETHER_API_KEY not found in environment")

    client = Together(api_key=api_key)
    messages = _build_messages(prompt, system_prompt)

    kwargs: Dict[str, Any] = {"model": model, "messages": messages, "temperature": temperature}
    if max_tokens:
        kwargs["max_tokens"] = max_tokens

    for attempt in range(MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(**kwargs)
            return _parse_response(response)
        except Exception as e:
            if not _should_retry(e, attempt):
                raise
            delay = _backoff(attempt)
            logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay:.2f}s...")
            time.sleep(delay)
    raise RuntimeError("Max retries exceeded")


async def acomplete(
    prompt: str,
    *,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
    system_prompt: Optional[str] = None,
) -> CompletionResponse:
    api_key = os.getenv("TOGETHER_API_KEY")
    if not api_key:
        raise ValueError("TOGETHER_API_KEY not found in environment")

    client = AsyncTogether(api_key=api_key)
    messages = _build_messages(prompt, system_prompt)

    kwargs: Dict[str, Any] = {"model": model, "messages": messages, "temperature": temperature}
    if max_tokens:
        kwargs["max_tokens"] = max_tokens

    # Apply rate limiting before each request
    rate_limiter = get_rate_limiter()

    for attempt in range(MAX_RETRIES + 1):
        try:
            await rate_limiter.acquire()
            response = await client.chat.completions.create(**kwargs)
            return _parse_response(response)
        except Exception as e:
            if not _should_retry(e, attempt):
                raise
            delay = _backoff(attempt)
            logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay:.2f}s...")
            await asyncio.sleep(delay)
    raise RuntimeError("Max retries exceeded")


async def batch_complete(
    prompts: List[str],
    *,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
    system_prompt: Optional[str] = None,
    max_concurrent: int = 10,
) -> List[CompletionResponse]:
    semaphore = asyncio.Semaphore(max_concurrent)

    async def _limited(prompt: str) -> CompletionResponse:
        async with semaphore:
            return await acomplete(prompt, model=model, temperature=temperature,
                                   max_tokens=max_tokens, system_prompt=system_prompt)

    return await asyncio.gather(*[_limited(p) for p in prompts])
