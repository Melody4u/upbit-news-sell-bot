import os
import logging


def setup_logger() -> None:
    """Log to file + console.

    Defaults:
    - File: logs/bot.log (INFO+)
    - Console (stderr): WARNING+ (so watchdog *.err.log contains only signals)

    Env:
    - LOG_LEVEL: root logger level (default INFO)
    - BOT_LOG_PATH: file path (default logs/bot.log)
    """
    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    root_level = getattr(logging, level_name, logging.INFO)

    log_path = os.getenv("BOT_LOG_PATH", "logs/bot.log")
    log_dir = os.path.dirname(log_path)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    root = logging.getLogger()
    # Avoid duplicate handlers when re-running in the same interpreter.
    root.handlers.clear()
    root.setLevel(root_level)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(fmt)

    stream_handler = logging.StreamHandler()  # stderr by default
    stream_handler.setLevel(logging.WARNING)
    stream_handler.setFormatter(fmt)

    root.addHandler(file_handler)
    root.addHandler(stream_handler)
