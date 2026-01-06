"""Configure application‑wide logging.

All modules in this package should import the `get_logger` function and
create a module‑specific logger to write messages to standard output.  The
format includes timestamps and log levels, aiding debugging and audit.
"""

import logging
from typing import Optional


def setup_root_logger(level: int = logging.INFO) -> None:
    """Configure the root logger with a basic format.

    Parameters
    ----------
    level: int, optional
        The minimum log level; defaults to `logging.INFO`.
    """

    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Return a logger with the specified name.

    Parameters
    ----------
    name: str, optional
        The name of the logger; if `None`, the root logger is returned.

    Returns
    -------
    logging.Logger
        A configured logger instance.
    """

    return logging.getLogger(name)


# Initialise logging on import
setup_root_logger()


if __name__ == "__main__":
    # Demonstrate logging
    logger = get_logger(__name__)
    logger.info("Logging is configured correctly.")