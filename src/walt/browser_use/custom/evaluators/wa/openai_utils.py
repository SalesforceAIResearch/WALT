"""Tools to generate from OpenAI prompts.
Adopted from https://github.com/zeno-ml/zeno-build/"""

import asyncio
import logging
import os
import random
import time
from typing import Any

import aiolimiter
import openai
from openai import OpenAIError, RateLimitError, APIError, AsyncOpenAI, OpenAI, AzureOpenAI
from tqdm.asyncio import tqdm_asyncio
from typing import Any, Union
from walt.browser_use.custom.evaluators.wa.constants import TOKEN_USAGE, SIMPLE_LLM_API_CACHE

logger = logging.getLogger('logger')


DISABLE_LLM_CACHE = os.getenv("DISABLE_LLM_CACHE", False)
logger.info(f"DISABLE_LLM_CACHE: {DISABLE_LLM_CACHE}")

def update_token_usage(model_name: str, token_stats: dict, token_usage_tracker: dict):
    # expect token_stats to include completion_tokens, prompt_tokens, num_requests
    if model_name not in token_usage_tracker:
        token_usage_tracker[model_name] = {
            'completion_tokens': 0,
            'prompt_tokens': 0,
            'num_requests': 0
        }

    token_usage_tracker[model_name]['completion_tokens'] += token_stats['completion_tokens']
    token_usage_tracker[model_name]['prompt_tokens'] += token_stats['prompt_tokens']
    token_usage_tracker[model_name]['num_requests'] += token_stats['num_requests']

    compl_token = token_usage_tracker[model_name]['completion_tokens']
    prompt_token = token_usage_tracker[model_name]['prompt_tokens']
    # may increment by alot
    req = token_usage_tracker[model_name]['num_requests']
    if req % 10 == 0 or token_stats['num_requests'] > 5:
        logger.info(
            f"[{model_name}] Avg. completion tokens: {compl_token/req:.2f}, prompt tokens: {prompt_token/req:.2f}")
        logger.info(
            f"[{model_name}] Total. completion tokens so far: {compl_token}, prompt tokens so far: {prompt_token}")
        logger.info(f"[{model_name}] Total. requests so far: {req}")
    return


def get_all_token_usage(token_usage_tracker: dict):
    all_token_usage = {}
    for m_name, m_stats in token_usage_tracker.items():
        all_token_usage[m_name] = {
            'completion_tokens': m_stats['completion_tokens'],
            'prompt_tokens': m_stats['prompt_tokens'],
            'num_requests': m_stats['num_requests']
        }
    return all_token_usage

def retry_with_exponential_backoff(  # type: ignore
    func,
    initial_delay: float = 1,
    exponential_base: float = 2,
    jitter: bool = True,
    max_retries: int = 3,
    errors: tuple[Any] = (RateLimitError,),
):
    """Retry a function with exponential backoff."""

    def wrapper(*args, **kwargs):  # type: ignore
        # Initialize variables
        num_retries = 0
        delay = initial_delay

        # Loop until a successful response or max_retries is hit or an exception is raised
        while True:
            try:
                return func(*args, **kwargs)
            # Retry on specified errors
            except errors as e:
                # Increment retries
                num_retries += 1

                # Check if max retries has been reached
                if num_retries > max_retries:
                    raise Exception(
                        f"Maximum number of retries ({max_retries}) exceeded."
                    )

                # Increment the delay
                delay *= exponential_base * (1 + jitter * random.random())
                print(f"Retrying in {delay} seconds.")
                # Sleep for the delay
                time.sleep(delay)

            # Raise exceptions for any errors not specified
            except Exception as e:
                raise e

    return wrapper


async def _throttled_openai_completion_acreate(
    engine: str,
    prompt: str,
    temperature: float,
    max_tokens: int,
    top_p: float,
    limiter: aiolimiter.AsyncLimiter,
) -> dict[str, Any]:
    async with limiter:
        for _ in range(3):
            try:
                return await openai.Completion.acreate(  # type: ignore
                    engine=engine,
                    prompt=prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                )
            except RateLimitError:
                logging.warning(
                    "OpenAI API rate limit exceeded. Sleeping for 10 seconds."
                )
                await asyncio.sleep(10)
            except APIError as e:
                logging.warning(f"OpenAI API error: {e}")
                break
        return {"choices": [{"message": {"content": ""}}]}


async def agenerate_from_openai_completion(
    prompts: list[str],
    engine: str,
    temperature: float,
    max_tokens: int,
    top_p: float,
    context_length: int,
    requests_per_minute: int = 300,
) -> list[str]:
    """Generate from OpenAI Completion API.

    Args:
        prompts: list of prompts
        temperature: Temperature to use.
        max_tokens: Maximum number of tokens to generate.
        top_p: Top p to use.
        context_length: Length of context to use.
        requests_per_minute: Number of requests per minute to allow.

    Returns:
        List of generated responses.
    """
    if "OPENAI_API_KEY" not in os.environ:
        raise ValueError(
            "OPENAI_API_KEY environment variable must be set when using OpenAI API."
        )
    openai.api_key = os.environ["OPENAI_API_KEY"]
    openai.organization = os.environ.get("OPENAI_ORGANIZATION", "")

    limiter = aiolimiter.AsyncLimiter(requests_per_minute)
    async_responses = [
        _throttled_openai_completion_acreate(
            engine=engine,
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            limiter=limiter,
        )
        for prompt in prompts
    ]
    responses = await tqdm_asyncio.gather(*async_responses)
    return [x["choices"][0]["text"] for x in responses]


@retry_with_exponential_backoff
def generate_from_openai_completion(
    prompt: str,
    engine: str,
    temperature: float,
    max_tokens: int,
    top_p: float,
    context_length: int,
    stop_token: str | None = None,
) -> str:
    if "OPENAI_API_KEY" not in os.environ:
        raise ValueError(
            "OPENAI_API_KEY environment variable must be set when using OpenAI API."
        )
    openai.api_key = os.environ["OPENAI_API_KEY"]
    openai.organization = os.environ.get("OPENAI_ORGANIZATION", "")
    response = openai.Completion.create(  # type: ignore
        prompt=prompt,
        engine=engine,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        stop=[stop_token],
    )
    answer: str = response["choices"][0]["text"]
    return answer


async def _throttled_openai_chat_completion_acreate(
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    max_tokens: int,
    top_p: float,
    limiter: aiolimiter.AsyncLimiter,
) -> dict[str, Any]:
    async with limiter:
        for _ in range(3):
            try:
                return await openai.ChatCompletion.acreate(  # type: ignore
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                )
            except RateLimitError:
                logging.warning(
                    "OpenAI API rate limit exceeded. Sleeping for 10 seconds."
                )
                await asyncio.sleep(10)
            except asyncio.exceptions.TimeoutError:
                logging.warning("OpenAI API timeout. Sleeping for 10 seconds.")
                await asyncio.sleep(10)
            except APIError as e:
                logging.warning(f"OpenAI API error: {e}")
                break
        return {"choices": [{"message": {"content": ""}}]}


async def agenerate_from_openai_chat_completion(
    messages_list: list[list[dict[str, str]]],
    engine: str,
    temperature: float,
    max_tokens: int,
    top_p: float,
    context_length: int,
    requests_per_minute: int = 300,
) -> list[str]:
    """Generate from OpenAI Chat Completion API.

    Args:
        messages_list: list of message list
        temperature: Temperature to use.
        max_tokens: Maximum number of tokens to generate.
        top_p: Top p to use.
        context_length: Length of context to use.
        requests_per_minute: Number of requests per minute to allow.

    Returns:
        List of generated responses.
    """
    if "OPENAI_API_KEY" not in os.environ:
        raise ValueError(
            "OPENAI_API_KEY environment variable must be set when using OpenAI API."
        )
    openai.api_key = os.environ["OPENAI_API_KEY"]
    openai.organization = os.environ.get("OPENAI_ORGANIZATION", "")

    limiter = aiolimiter.AsyncLimiter(requests_per_minute)
    async_responses = [
        _throttled_openai_chat_completion_acreate(
            model=engine,
            messages=message,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            limiter=limiter,
        )
        for message in messages_list
    ]
    responses = await tqdm_asyncio.gather(*async_responses)
    return [x["choices"][0]["message"]["content"] for x in responses]

def _completion_args_to_cache_key(
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    max_tokens: int,
    top_p: float,
    num_outputs: int = 1,
) -> str:
    return f"{model}_{messages}_{temperature}_{max_tokens}_{top_p}_{num_outputs}"

@retry_with_exponential_backoff
def generate_from_openai_chat_completion(
    client: OpenAI,
    messages: list[dict[str, str]],
    model: str,
    temperature: float,
    max_tokens: int,
    top_p: float,
    context_length: int,
    stop_token: str | None = None,
    num_outputs: int = 1,
) -> Union[str, list[str]]:
    cache_key = _completion_args_to_cache_key(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        num_outputs=num_outputs
    )
    if not DISABLE_LLM_CACHE and cache_key in SIMPLE_LLM_API_CACHE:
        logger.info(f"generate_from_openai_chat_completion hit cache")
        return SIMPLE_LLM_API_CACHE[cache_key]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        n=num_outputs
    )

    token_stats = {
        'completion_tokens': response.usage.completion_tokens,
        'prompt_tokens': response.usage.prompt_tokens,
        'num_requests': 1
    }
    update_token_usage(
        model_name=model,
        token_stats=token_stats,
        token_usage_tracker=TOKEN_USAGE
    )

    if num_outputs > 1:
        answer: list[str] = [x.message.content for x in response.choices]
    else:
        answer: str = response.choices[0].message.content

    SIMPLE_LLM_API_CACHE[cache_key] = answer
    return answer


@retry_with_exponential_backoff
# debug only
def fake_generate_from_openai_chat_completion(
    messages: list[dict[str, str]],
    model: str,
    temperature: float,
    max_tokens: int,
    top_p: float,
    context_length: int,
    stop_token: str | None = None,
) -> str:
    if "OPENAI_API_KEY" not in os.environ:
        raise ValueError(
            "OPENAI_API_KEY environment variable must be set when using OpenAI API."
        )
    openai.api_key = os.environ["OPENAI_API_KEY"]
    openai.organization = os.environ.get("OPENAI_ORGANIZATION", "")
    answer = "Let's think step-by-step. This page shows a list of links and buttons. There is a search box with the label 'Search query'. I will click on the search box to type the query. So the action I will perform is \"click [60]\"."
    return answer
