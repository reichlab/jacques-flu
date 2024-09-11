import structlog

from jacques_flu.util.date import get_current_date
from jacques_flu.util.logs import setup_logging

#innocuous comment

setup_logging()
logger = structlog.get_logger()


def main():
    """Application entry point."""

    today = get_current_date()
    logger.info("retrieved the date", date=today)

    return f"Hello, today is {today}!"


if __name__ == "__main__":
    main()
