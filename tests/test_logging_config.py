"""
Tests for the logging configuration module.

This test suite validates:
- ProgressLogger for tracking progress in logs
- setup_logging configuration
- Structured test result logging
- Context-aware logging
"""
import unittest
import logging
import os
import tempfile
from unittest.mock import patch, MagicMock, call
from datetime import datetime, timedelta
from klareco.logging_config import (
    ProgressLogger,
    setup_logging,
    log_test_result,
    log_with_context
)


class TestProgressLogger(unittest.TestCase):
    """Test suite for the ProgressLogger class."""

    def setUp(self):
        """Set up a mock logger for testing."""
        self.mock_logger = MagicMock()

    def test_init_creates_progress_logger_with_defaults(self):
        """Tests that ProgressLogger initializes with correct defaults."""
        progress = ProgressLogger(total=100, desc="Test Progress")

        self.assertEqual(progress.total, 100)
        self.assertEqual(progress.desc, "Test Progress")
        self.assertEqual(progress.current, 0)
        self.assertEqual(progress.last_log_percent, -1)
        self.assertIsNotNone(progress.start_time)

    def test_init_uses_provided_logger(self):
        """Tests that ProgressLogger uses the provided logger."""
        progress = ProgressLogger(total=100, logger=self.mock_logger)

        self.assertEqual(progress.logger, self.mock_logger)

    def test_update_increments_current_count(self):
        """Tests that update() increments the current count."""
        progress = ProgressLogger(total=100, logger=self.mock_logger)

        progress.update(5)
        self.assertEqual(progress.current, 5)

        progress.update(10)
        self.assertEqual(progress.current, 15)

    def test_update_logs_at_10_percent_intervals(self):
        """Tests that update() logs progress at 10% intervals."""
        progress = ProgressLogger(total=100, logger=self.mock_logger)

        # Update to 5% - should not log (below 10%)
        progress.update(5)
        self.assertEqual(self.mock_logger.info.call_count, 0)

        # Update to 15% - should log (crossed 10%)
        progress.update(10)
        self.assertEqual(self.mock_logger.info.call_count, 1)

        # Update to 18% - should not log (still within 10-20% range)
        progress.update(3)
        self.assertEqual(self.mock_logger.info.call_count, 1)

        # Update to 25% - should log (crossed 20%)
        progress.update(7)
        self.assertEqual(self.mock_logger.info.call_count, 2)

    def test_update_logs_when_item_description_provided(self):
        """Tests that update() logs immediately when item description is provided."""
        progress = ProgressLogger(total=100, logger=self.mock_logger)

        # Update with description at 5% - should log despite being below 10%
        progress.update(5, item_desc="Processing item 1")
        self.assertEqual(self.mock_logger.info.call_count, 1)

        # Verify description is in the logged message
        logged_message = self.mock_logger.info.call_args[0][0]
        self.assertIn("Processing item 1", logged_message)

    def test_update_logs_progress_percentage(self):
        """Tests that update() includes progress percentage in log message."""
        progress = ProgressLogger(total=100, desc="Test", logger=self.mock_logger)

        progress.update(25)  # 25%

        logged_message = self.mock_logger.info.call_args[0][0]
        self.assertIn("Test: 25/100 (25%)", logged_message)

    def test_update_calculates_eta(self):
        """Tests that update() calculates and logs ETA."""
        progress = ProgressLogger(total=100, logger=self.mock_logger)

        # Manually set start_time to ensure predictable ETA calculation
        progress.start_time = datetime.now() - timedelta(seconds=10)
        progress.update(50)  # 50% complete in 10 seconds

        logged_message = self.mock_logger.info.call_args[0][0]
        # ETA should be present for incomplete progress
        self.assertIn("ETA:", logged_message)

    def test_update_no_eta_when_complete(self):
        """Tests that update() does not show ETA when progress is complete."""
        progress = ProgressLogger(total=100, logger=self.mock_logger)

        progress.update(100)  # 100% complete

        logged_message = self.mock_logger.info.call_args[0][0]
        # ETA should NOT be present when complete
        self.assertNotIn("ETA:", logged_message)

    def test_update_handles_zero_total(self):
        """Tests that update() handles zero total without division error."""
        progress = ProgressLogger(total=0, logger=self.mock_logger)

        # Should not raise ZeroDivisionError
        # Force logging by providing item_desc
        progress.update(1, item_desc="Test item")

        # Percent should be 0
        logged_message = self.mock_logger.info.call_args[0][0]
        self.assertIn("(0%)", logged_message)

    def test_close_marks_progress_complete(self):
        """Tests that close() marks progress as 100% complete."""
        progress = ProgressLogger(total=100, logger=self.mock_logger)

        progress.update(50)
        progress.close()

        # Current should be set to total
        self.assertEqual(progress.current, 100)

        # Should have logged completion
        self.assertGreater(self.mock_logger.info.call_count, 0)

    def test_close_does_nothing_if_already_complete(self):
        """Tests that close() is idempotent when already complete."""
        progress = ProgressLogger(total=100, logger=self.mock_logger)

        progress.update(100)  # Already at 100%
        call_count_before = self.mock_logger.info.call_count

        progress.close()

        # Should not log again (already complete)
        self.assertEqual(self.mock_logger.info.call_count, call_count_before)


class TestSetupLogging(unittest.TestCase):
    """Test suite for the setup_logging() function."""

    def setUp(self):
        """Clear root logger handlers before each test."""
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

    def tearDown(self):
        """Clean up handlers after each test."""
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

    def test_setup_logging_creates_file_handler(self):
        """Tests that setup_logging creates a file handler."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            log_file = tmp_file.name

        try:
            setup_logging(log_file=log_file)

            root_logger = logging.getLogger()
            file_handlers = [h for h in root_logger.handlers if isinstance(h, logging.FileHandler)]

            self.assertEqual(len(file_handlers), 1)
        finally:
            # Clean up
            if os.path.exists(log_file):
                os.remove(log_file)

    def test_setup_logging_creates_console_handler(self):
        """Tests that setup_logging creates a console (StreamHandler) handler."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            log_file = tmp_file.name

        try:
            setup_logging(log_file=log_file)

            root_logger = logging.getLogger()
            stream_handlers = [h for h in root_logger.handlers if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)]

            self.assertEqual(len(stream_handlers), 1)
        finally:
            if os.path.exists(log_file):
                os.remove(log_file)

    def test_setup_logging_sets_info_level_by_default(self):
        """Tests that setup_logging sets INFO level by default."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            log_file = tmp_file.name

        try:
            setup_logging(log_file=log_file)

            root_logger = logging.getLogger()
            self.assertEqual(root_logger.level, logging.INFO)
        finally:
            if os.path.exists(log_file):
                os.remove(log_file)

    def test_setup_logging_sets_debug_level_when_debug_true(self):
        """Tests that setup_logging sets DEBUG level when debug=True."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            log_file = tmp_file.name

        try:
            setup_logging(log_file=log_file, debug=True)

            root_logger = logging.getLogger()
            self.assertEqual(root_logger.level, logging.DEBUG)
        finally:
            if os.path.exists(log_file):
                os.remove(log_file)

    def test_setup_logging_uses_enhanced_format_in_debug_mode(self):
        """Tests that setup_logging uses enhanced format string in debug mode."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            log_file = tmp_file.name

        try:
            setup_logging(log_file=log_file, debug=True)

            root_logger = logging.getLogger()
            # Check that handlers have enhanced format (includes filename and lineno)
            for handler in root_logger.handlers:
                format_string = handler.formatter._fmt
                self.assertIn("%(filename)s", format_string)
                self.assertIn("%(lineno)d", format_string)
        finally:
            if os.path.exists(log_file):
                os.remove(log_file)

    def test_setup_logging_uses_simple_format_in_normal_mode(self):
        """Tests that setup_logging uses simple format string in normal mode."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            log_file = tmp_file.name

        try:
            setup_logging(log_file=log_file, debug=False)

            root_logger = logging.getLogger()
            # Check that handlers have simple format (no filename/lineno)
            for handler in root_logger.handlers:
                format_string = handler.formatter._fmt
                self.assertNotIn("%(filename)s", format_string)
                self.assertNotIn("%(lineno)d", format_string)
        finally:
            if os.path.exists(log_file):
                os.remove(log_file)

    def test_setup_logging_writes_run_separator(self):
        """Tests that setup_logging writes a run separator to the log."""
        with tempfile.NamedTemporaryFile(delete=False, mode='w') as tmp_file:
            log_file = tmp_file.name

        try:
            setup_logging(log_file=log_file)

            # Read the log file
            with open(log_file, 'r') as f:
                log_content = f.read()

            # Verify run separator is present
            self.assertIn("=" * 80, log_content)
            self.assertIn("NEW RUN STARTED", log_content)
        finally:
            if os.path.exists(log_file):
                os.remove(log_file)

    def test_setup_logging_clears_existing_handlers(self):
        """Tests that setup_logging clears existing handlers to avoid duplicates."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            log_file = tmp_file.name

        try:
            # Add a dummy handler
            root_logger = logging.getLogger()
            dummy_handler = logging.StreamHandler()
            root_logger.addHandler(dummy_handler)
            initial_count = len(root_logger.handlers)

            # Call setup_logging
            setup_logging(log_file=log_file)

            # Should have exactly 2 handlers (file + console), not initial_count + 2
            self.assertEqual(len(root_logger.handlers), 2)
        finally:
            if os.path.exists(log_file):
                os.remove(log_file)


class TestLogTestResult(unittest.TestCase):
    """Test suite for the log_test_result() function."""

    @patch('klareco.logging_config.logging.getLogger')
    def test_log_test_result_pass_uses_info_level(self, mock_get_logger):
        """Tests that PASS status uses INFO log level."""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        log_test_result("test_example", "PASS")

        # Verify logger.log was called with INFO level
        mock_logger.log.assert_called_once()
        self.assertEqual(mock_logger.log.call_args[0][0], logging.INFO)

    @patch('klareco.logging_config.logging.getLogger')
    def test_log_test_result_fail_uses_error_level(self, mock_get_logger):
        """Tests that FAIL status uses ERROR log level."""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        log_test_result("test_example", "FAIL")

        # Verify logger.log was called with ERROR level
        mock_logger.log.assert_called_once()
        self.assertEqual(mock_logger.log.call_args[0][0], logging.ERROR)

    @patch('klareco.logging_config.logging.getLogger')
    def test_log_test_result_skip_uses_warning_level(self, mock_get_logger):
        """Tests that SKIP status uses WARNING log level."""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        log_test_result("test_example", "SKIP")

        # Verify logger.log was called with WARNING level
        mock_logger.log.assert_called_once()
        self.assertEqual(mock_logger.log.call_args[0][0], logging.WARNING)

    @patch('klareco.logging_config.logging.getLogger')
    def test_log_test_result_includes_test_name(self, mock_get_logger):
        """Tests that log message includes the test name."""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        log_test_result("test_my_feature", "PASS")

        logged_message = mock_logger.log.call_args[0][1]
        self.assertIn("test_my_feature", logged_message)

    @patch('klareco.logging_config.logging.getLogger')
    def test_log_test_result_includes_duration_when_provided(self, mock_get_logger):
        """Tests that log message includes duration when provided."""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        log_test_result("test_example", "PASS", duration_ms=123.45)

        logged_message = mock_logger.log.call_args[0][1]
        self.assertIn("(123ms)", logged_message)

    @patch('klareco.logging_config.logging.getLogger')
    def test_log_test_result_includes_error_when_provided(self, mock_get_logger):
        """Tests that log message includes error message when provided."""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        log_test_result("test_example", "FAIL", error="Expected 5, got 3")

        logged_message = mock_logger.log.call_args[0][1]
        self.assertIn("Expected 5, got 3", logged_message)

    @patch('klareco.logging_config.logging.getLogger')
    def test_log_test_result_uses_correct_symbols(self, mock_get_logger):
        """Tests that log messages use correct symbols for each status."""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        # Test PASS symbol (✓)
        log_test_result("test_pass", "PASS")
        pass_message = mock_logger.log.call_args[0][1]
        self.assertIn("✓", pass_message)

        # Test FAIL symbol (✗)
        mock_logger.reset_mock()
        log_test_result("test_fail", "FAIL")
        fail_message = mock_logger.log.call_args[0][1]
        self.assertIn("✗", fail_message)

        # Test SKIP symbol (○)
        mock_logger.reset_mock()
        log_test_result("test_skip", "SKIP")
        skip_message = mock_logger.log.call_args[0][1]
        self.assertIn("○", skip_message)


class TestLogWithContext(unittest.TestCase):
    """Test suite for the log_with_context() function."""

    @patch('klareco.logging_config.logging.getLogger')
    def test_log_with_context_logs_main_message(self, mock_get_logger):
        """Tests that log_with_context logs the main message."""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        log_with_context("Processing user input", level=logging.INFO)

        # Verify main message was logged
        mock_logger.log.assert_called()
        self.assertEqual(mock_logger.log.call_args[0][0], logging.INFO)
        self.assertEqual(mock_logger.log.call_args[0][1], "Processing user input")

    @patch('klareco.logging_config.logging.getLogger')
    def test_log_with_context_logs_context_when_debug_enabled(self, mock_get_logger):
        """Tests that log_with_context logs context dict when DEBUG is enabled."""
        mock_logger = MagicMock()
        mock_logger.isEnabledFor.return_value = True  # DEBUG enabled
        mock_get_logger.return_value = mock_logger

        context = {"input": "test data", "state": "processing"}
        log_with_context("Test message", context=context, level=logging.DEBUG)

        # Verify main message was logged
        self.assertGreaterEqual(mock_logger.log.call_count, 1)

        # Verify context items were logged with debug
        debug_calls = [call for call in mock_logger.debug.call_args_list]
        self.assertGreater(len(debug_calls), 0)

        # Verify context values are in debug messages
        all_debug_messages = " ".join([str(call) for call in debug_calls])
        self.assertIn("input", all_debug_messages)
        self.assertIn("test data", all_debug_messages)
        self.assertIn("state", all_debug_messages)
        self.assertIn("processing", all_debug_messages)

    @patch('klareco.logging_config.logging.getLogger')
    def test_log_with_context_skips_context_when_debug_disabled(self, mock_get_logger):
        """Tests that log_with_context skips context when DEBUG is disabled."""
        mock_logger = MagicMock()
        mock_logger.isEnabledFor.return_value = False  # DEBUG disabled
        mock_get_logger.return_value = mock_logger

        context = {"input": "test data", "state": "processing"}
        log_with_context("Test message", context=context, level=logging.INFO)

        # Verify main message was logged
        mock_logger.log.assert_called()

        # Verify no debug calls were made (context skipped)
        self.assertEqual(mock_logger.debug.call_count, 0)

    @patch('klareco.logging_config.logging.getLogger')
    def test_log_with_context_truncates_long_values(self, mock_get_logger):
        """Tests that log_with_context truncates values longer than 200 chars."""
        mock_logger = MagicMock()
        mock_logger.isEnabledFor.return_value = True
        mock_get_logger.return_value = mock_logger

        long_value = "x" * 300
        context = {"data": long_value}
        log_with_context("Test message", context=context, level=logging.DEBUG)

        # Find the debug call with the truncated value
        debug_calls = mock_logger.debug.call_args_list
        self.assertGreater(len(debug_calls), 0)

        # Verify truncation occurred (should end with "...")
        last_debug_message = debug_calls[-1][0][0]
        self.assertIn("...", last_debug_message)
        # Verify message is not 300+ chars (should be truncated to ~200)
        self.assertLess(len(last_debug_message), 250)

    @patch('klareco.logging_config.logging.getLogger')
    def test_log_with_context_handles_none_context(self, mock_get_logger):
        """Tests that log_with_context handles None context gracefully."""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        # Should not raise error
        log_with_context("Test message", context=None, level=logging.INFO)

        # Verify main message was logged
        mock_logger.log.assert_called_once()


if __name__ == '__main__':
    unittest.main()
