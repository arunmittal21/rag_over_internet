import logging
from logging.handlers import RotatingFileHandler

def setup_logger(
    log_file: str = "tmp/llm_debug.log",
    max_bytes: int = 1 * 1024 * 1024, 
    backup_count: int = 5,
    level: int = logging.DEBUG
):
    """
    Set up a rotating logger for LangChain and LLM HTTP calls.
    """
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    handler = RotatingFileHandler(
        log_file, maxBytes=max_bytes, backupCount=backup_count
    )
    handler.setFormatter(formatter)

    # Apply to root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.addHandler(handler)

    # LangChain and related libraries
    logging.getLogger("langchain").setLevel(logging.DEBUG)
    logging.getLogger("httpx").setLevel(logging.DEBUG)
    logging.getLogger("httpcore").setLevel(logging.DEBUG)
    logging.getLogger("aiohttp").setLevel(logging.DEBUG)

    # Suppress overly verbose logs
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)
