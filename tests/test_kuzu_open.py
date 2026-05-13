"""Tests for the centralized Kuzu opener (klareco/utils/kuzu_open.py)."""

import unittest
from unittest import mock

from klareco.utils import kuzu_open
from klareco.utils.kuzu_open import open_kuzu


class TestKuzuOpenKwargs(unittest.TestCase):
    """Verify open_kuzu passes the right kwargs to kuzu.Database."""

    def setUp(self):
        # Patch kuzu.Database inside the helper's module so we can
        # inspect the kwargs without touching disk.
        patcher = mock.patch.object(kuzu_open, 'kuzu')
        self.mock_kuzu = patcher.start()
        self.addCleanup(patcher.stop)

        # Clear env vars for a clean baseline; restore after each test.
        env_patcher = mock.patch.dict('os.environ', clear=False)
        env_patcher.start()
        self.addCleanup(env_patcher.stop)
        import os
        os.environ.pop('KLARECO_KUZU_BUFFER_MB', None)
        os.environ.pop('KLARECO_KUZU_MAX_THREADS', None)

    def _kwargs_passed_to_database(self):
        self.mock_kuzu.Database.assert_called_once()
        return self.mock_kuzu.Database.call_args.kwargs

    def test_default_kwargs(self):
        """No env vars set → only read_only is passed.

        We deliberately do NOT cap buffer_pool_size by default: the
        production graph needs Kuzu's 80%-of-RAM default for multi-hop
        traversals. Parallel workloads that need a tighter per-worker
        budget set the env var.
        """
        open_kuzu('/tmp/fake.db')
        kwargs = self._kwargs_passed_to_database()
        self.assertTrue(kwargs['read_only'])
        self.assertNotIn('buffer_pool_size', kwargs)
        self.assertNotIn('max_num_threads', kwargs)

    def test_buffer_env_var_propagates(self):
        import os
        os.environ['KLARECO_KUZU_BUFFER_MB'] = '256'
        open_kuzu('/tmp/fake.db')
        self.assertEqual(
            self._kwargs_passed_to_database()['buffer_pool_size'],
            256 * 1024 * 1024,
        )

    def test_threads_env_var_propagates(self):
        import os
        os.environ['KLARECO_KUZU_MAX_THREADS'] = '4'
        open_kuzu('/tmp/fake.db')
        self.assertEqual(self._kwargs_passed_to_database()['max_num_threads'], 4)

    def test_read_only_can_be_disabled(self):
        open_kuzu('/tmp/fake.db', read_only=False)
        self.assertFalse(self._kwargs_passed_to_database()['read_only'])

    def test_falls_back_when_kuzu_rejects_kwargs(self):
        """Older Kuzu versions raise TypeError on unknown kwargs."""
        self.mock_kuzu.Database.side_effect = [TypeError('unsupported kw'), mock.MagicMock()]
        open_kuzu('/tmp/fake.db')
        self.assertEqual(self.mock_kuzu.Database.call_count, 2)
        # Second call has no kwargs (positional path arg only).
        second_call = self.mock_kuzu.Database.call_args_list[1]
        self.assertEqual(second_call.kwargs, {})


if __name__ == '__main__':
    unittest.main()
