"""Caching utilities for expensive computations.

This module previously wrapped Streamlit's caching decorators (`st.cache_data`
and `st.cache_resource`) so that expensive functions could be memoised both
inside and outside Streamlit.  In order to simplify the codebase and remove
any reliance on hidden caching mechanisms, the decorators defined here now
behave as identity functions.  That is, they simply return the original
function without applying any caching.  Removing caching
avoids the need for hashing and makes the implementation easier to
understand for educational purposes.
"""

from typing import Callable, TypeVar

F = TypeVar("F", bound=Callable[..., object])


def cache_data(func: F) -> F:
    """Identity decorator for data caching.

    This decorator previously enabled caching of function results.  It now
    simply returns the input function unchanged so that caching is
    effectively disabled.
    """
    return func


def cache_resource(func: F) -> F:
    """Identity decorator for resource caching.

    This decorator previously cached resources across reruns when using
    Streamlit.  It now returns the function unmodified.
    """
    return func


if __name__ == "__main__":
    # Demonstrate identity behaviour
    @cache_data
    def expensive(x: int) -> int:
        print(f"Computing square of {x}")
        return x * x

    print(expensive(3))
    print(expensive(3))  # Without caching this prints twice
