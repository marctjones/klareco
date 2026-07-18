"""
BiographyFormatStage: when the user asks "tell me about X" or "what is X",
replace the one-sentence extractor output with a multi-sentence
paragraph from klareco.generation (#766 #775 orchestrator wiring).

Runs after ExtractAndGenerate. If the question matches a
"diru-pri-X" or "kio-estas-X" pattern AND the entity has facts in
entity_facts, replace final_text with the generated paragraph.
Otherwise leave the extractor's output alone.
"""
from __future__ import annotations

import logging
import re
from typing import Optional

from klareco.orchestrator.context import QueryContext, ContextDelta
from klareco.orchestrator.dependencies import TableDependency
from klareco.orchestrator.stage import PipelineStage
from klareco.generation import biography, define

logger = logging.getLogger(__name__)


# Patterns the user might use to ask for a paragraph-level answer
_BIOGRAPHY_TRIGGERS = (
    r'\bdiru\s+(?:al\s+mi\s+)?pri\s+(\w+)',
    r'\brakontu\s+(?:al\s+mi\s+)?pri\s+(\w+)',
    r'\bkiu\s+estas\s+(\w+)\s*\?',
    r'\bkio\s+estas\s+(\w+)\s*\?',
)

_DEFINE_TRIGGERS = (
    r'\bkio\s+estas\s+(\w+)\s*\?',
    r'\bdifinu\s+(\w+)',
)


def _extract_entity_from_question(question: str) -> Optional[tuple[str, str]]:
    """Return (entity_radiko, intent) where intent ∈ {biography, define}."""
    q = (question or '').lower()
    for pat in _BIOGRAPHY_TRIGGERS:
        m = re.search(pat, q)
        if m:
            return m.group(1), 'biography'
    for pat in _DEFINE_TRIGGERS:
        m = re.search(pat, q)
        if m:
            return m.group(1), 'define'
    return None


class BiographyFormatStage(PipelineStage):
    name = 'biography_format'

    # Loud-failure contract (#884). As of #881 generation reads entity_facts
    # through the SLOTS adapter over the live TRIPLE schema, so the real
    # requirement is the triple columns. (Stays default-off — the facts are too
    # thin/noisy to generate a biography, tracked by #745 — but if enabled it
    # now passes preflight and degrades gracefully ("Mi ne havas faktojn…")
    # instead of crashing.)
    REQUIRES = (
        TableDependency('entity_facts',
                        columns=('sid', 'entito', 'rilato', 'valoro'),
                        issue='#745'),
    )

    def should_skip(self, ctx: QueryContext) -> bool:
        # Short-circuited by a tool? Skip.
        if ctx.flag('tool_short_circuit'):
            return True
        # No question text? Skip.
        return not ctx.question

    def run(self, ctx: QueryContext) -> ContextDelta:
        match = _extract_entity_from_question(ctx.question)
        if match is None:
            return ContextDelta()
        raw_entity, intent = match
        # Try several radiko candidates: as-is, accusative-stripped,
        # noun-ending-stripped. The first that produces a non-empty
        # generation wins. This handles 'Esperanton' (acc) → 'esperant'
        # as well as 'Kurosawa' (already a bare name) without
        # over-stripping the 'a'.
        candidates = [raw_entity]
        if raw_entity.endswith('n'):
            candidates.append(raw_entity[:-1])      # acc 'Esperanton' → 'Esperanto'
            candidates.append(raw_entity[:-2])      # 'Esperanton' → 'Esperant'
        elif raw_entity.endswith('on'):
            candidates.append(raw_entity[:-2])
        if raw_entity.endswith(('o', 'a', 'e', 'i')):
            candidates.append(raw_entity[:-1])
        text = ''
        used = raw_entity
        try:
            for c in candidates:
                if intent == 'biography':
                    text = biography(c)
                else:
                    text = define(c)
                if text and not text.startswith('Mi ne havas'):
                    used = c
                    break
                text = ''
        except Exception as e:
            logger.warning(f'[biography_format] generation failed for '
                           f'{raw_entity!r}: {e}')
            return ContextDelta()
        if not text:
            return ContextDelta()
        entity_radiko = used
        logger.info(f'[biography_format] generated {intent} for {entity_radiko!r}')
        return ContextDelta(
            symbolic={'final_text': text},
            flags={'biography_format_applied': True,
                   'biography_entity': entity_radiko,
                   'biography_intent': intent},
        )

    # on_failure deliberately NOT overridden (#884): this stage is default-off
    # until #881 lands; when explicitly enabled, a failure must be LOUD.
    # (The old override swallowed the BinderException that hid #881 for weeks.)
