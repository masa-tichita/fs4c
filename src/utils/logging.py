import asyncio
import sys
from collections.abc import Callable
from functools import wraps
from time import time
from typing import ParamSpec, TypeVar, cast

from loguru import logger

P = ParamSpec("P")
R = TypeVar("R")


def log_fn(fn: Callable[P, R]) -> Callable[P, R]:
    if asyncio.iscoroutinefunction(fn):

        async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            start_time = time()
            result = await fn(*args, **kwargs)
            end_time = time()
            txt = f"Called {fn.__name__} with elapsed_time={round(end_time - start_time, 3)} "
            logger.opt(depth=1).info(txt)
            return result

        return cast(Callable[P, R], async_wrapper)
    else:

        @wraps(fn)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            start_time = time()
            result = fn(*args, **kwargs)
            end_time = time()
            txt = f"Called {fn.__name__} with elapsed_time={round(end_time - start_time, 3)} "
            logger.opt(depth=1).info(txt)
            return result

        return cast(Callable[P, R], wrapper)


def debug_fn(fn: Callable[P, R]) -> Callable[P, R]:
    if asyncio.iscoroutinefunction(fn):

        async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            start_time = time()
            result = await fn(*args, **kwargs)
            end_time = time()
            txt = f"Called {fn.__name__} with elapsed_time={round(end_time - start_time, 3)} "
            logger.opt(depth=1).debug(txt)
            return result

        return cast(Callable[P, R], async_wrapper)
    else:

        @wraps(fn)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            start_time = time()
            result = fn(*args, **kwargs)
            end_time = time()
            txt = f"Called {fn.__name__} with elapsed_time={round(end_time - start_time, 3)} "
            logger.opt(depth=1).debug(txt)
            return result

        return cast(Callable[P, R], wrapper)


def setup_logger():
    logger.remove()
    logger.add(
        sys.stdout,
        level="DEBUG",
    )
    logger.info("Logger setup")
