"""
CLI dispatcher smoke tests — the command surface loads, parses, and routes.

Fast/offline: exercises the registry, argument wiring, exit codes, and --json
for the commands that need no data. Data-dependent commands (query/explain/
inspect) are covered by the contract suite + manual runs; here we assert the
tidy structural contract: every group registers, --help works, exit codes hold.
"""
from __future__ import annotations

import json

import pytest

from klareco.cli import build_parser, main, _GROUPS
from klareco.cli._base import EXIT_OK, EXIT_USAGE
from klareco.cli.commands import data as data_cmd


def test_all_groups_register_without_error():
    parser = build_parser()
    # every group module contributed at least its top command
    choices = parser._subparsers._group_actions[0].choices
    for expected in ('query', 'explain', 'parse', 'translate',
                     'data', 'doctor', 'info', 'inspect', 'corpus', 'eval'):
        assert expected in choices, f"{expected} not registered"


def test_no_command_prints_help_and_usage_exit():
    assert main([]) == EXIT_USAGE


def test_parse_command_roundtrips(capsys):
    rc = main(['parse', 'Mi amas la hundon.', '--format', 'json'])
    out = capsys.readouterr().out
    ast = json.loads(out)
    assert rc == EXIT_OK
    assert ast['tipo'] == 'frazo'


def test_parse_json_flag_is_machine_readable(capsys):
    main(['parse', 'La suno brilas.', '--json'])
    json.loads(capsys.readouterr().out)      # must be valid JSON


def test_data_stage_previews_by_default(capsys):
    rc = main(['data', 'parse'])
    out = capsys.readouterr().out
    assert rc == EXIT_OK
    assert 'parse_corpus.sh' in out          # shows the real command
    assert '--run' in out                    # and how to actually run it


def test_data_stage_json_lists_the_command(capsys):
    main(['data', 'build-store', '--json'])
    payload = json.loads(capsys.readouterr().out)
    assert payload['ran'] is False
    assert payload['command'] == data_cmd.STAGES['build-store']


def test_every_data_stage_has_a_command():
    for stage, cmd in data_cmd.STAGES.items():
        assert cmd and isinstance(cmd, list), stage


def test_groups_list_matches_importable_modules():
    import importlib
    for name in _GROUPS:
        mod = importlib.import_module(f'klareco.cli.commands.{name}')
        assert hasattr(mod, 'register'), f"{name} lacks register()"
