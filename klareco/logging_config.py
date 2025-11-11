import logging
import os
import sys
from datetime import datetime


class ProgressLogger:
    """
    A logger that shows progress information in both console and log file.
    Works as a replacement for tqdm for watch.sh visibility.
    """
    def __init__(self, total, desc="Progress", logger=None):
        self.total = total
        self.current = 0
        self.desc = desc
        self.logger = logger or logging.getLogger()
        self.start_time = datetime.now()
        self.last_log_percent = -1

    def update(self, n=1, item_desc=None):
        """Update progress by n items."""
        self.current += n
        percent = int((self.current / self.total) * 100) if self.total > 0 else 0

        # Log every 10% or when description changes
        if percent - self.last_log_percent >= 10 or item_desc or self.current == self.total:
            elapsed = (datetime.now() - self.start_time).total_seconds()
            rate = self.current / elapsed if elapsed > 0 else 0
            eta_seconds = (self.total - self.current) / rate if rate > 0 else 0

            msg_parts = [f"{self.desc}: {self.current}/{self.total} ({percent}%)"]
            if item_desc:
                msg_parts.append(f"- {item_desc}")
            if eta_seconds > 0 and self.current < self.total:
                msg_parts.append(f"[ETA: {int(eta_seconds)}s]")

            self.logger.info(" ".join(msg_parts))
            self.last_log_percent = percent

    def close(self):
        """Mark progress as complete."""
        if self.current < self.total:
            self.current = self.total
            self.update(0)


def setup_logging(log_file='klareco.log', level=logging.INFO, debug=False):
    """
    Set up comprehensive logging for the application.

    Args:
        log_file: Path to the log file. Defaults to 'klareco.log'.
        level: Logging level (default: INFO). Use DEBUG for verbose output.
        debug: If True, enables DEBUG level with extra context.
    """
    # Clear existing handlers to avoid duplicates
    root_logger = logging.getLogger()
    if root_logger.handlers:
        for handler in root_logger.handlers:
            root_logger.removeHandler(handler)

    # Set level based on debug flag
    if debug:
        level = logging.DEBUG

    # Enhanced format with more context for debugging
    if debug:
        format_string = '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
    else:
        format_string = '%(asctime)s - %(levelname)s - %(message)s'

    # Create handlers
    file_handler = logging.FileHandler(log_file, mode='a')
    file_handler.setLevel(level)
    file_handler.setFormatter(logging.Formatter(format_string))

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(logging.Formatter(format_string))

    # Configure root logger
    root_logger.setLevel(level)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    # Add run separator
    logging.info("=" * 80)
    logging.info(f"NEW RUN STARTED - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    if debug:
        logging.info("DEBUG MODE ENABLED - Verbose logging active")
    logging.info("=" * 80)


def log_test_result(test_name, status, duration_ms=None, error=None):
    """
    Log a test result in a structured format.

    Args:
        test_name: Name of the test
        status: 'PASS', 'FAIL', or 'SKIP'
        duration_ms: Optional duration in milliseconds
        error: Optional error message if failed
    """
    logger = logging.getLogger()

    symbol = {"PASS": "✓", "FAIL": "✗", "SKIP": "○"}.get(status, "?")
    level = logging.INFO if status == "PASS" else logging.ERROR if status == "FAIL" else logging.WARNING

    msg_parts = [f"TEST {symbol} {status}:", test_name]
    if duration_ms is not None:
        msg_parts.append(f"({duration_ms:.0f}ms)")
    if error:
        msg_parts.append(f"- {error}")

    logger.log(level, " ".join(msg_parts))


def log_with_context(message, context=None, level=logging.DEBUG):
    """
    Log a message with additional context (inputs, state, etc.).

    Args:
        message: Main log message
        context: Dict of contextual information
        level: Log level (default: DEBUG)
    """
    logger = logging.getLogger()
    logger.log(level, message)

    if context and logger.isEnabledFor(logging.DEBUG):
        for key, value in context.items():
            # Truncate long values
            str_value = str(value)
            if len(str_value) > 200:
                str_value = str_value[:200] + "..."
            logger.debug(f"  └─ {key}: {str_value}")