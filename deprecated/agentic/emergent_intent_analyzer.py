"""
Emergent Intent Analyzer - Detect new intent patterns from execution traces

Part of Phase 9: Learning Loop
Identifies query patterns that:
- Are not well-handled by existing experts
- Occur frequently enough to warrant new specialized handling
- Show consistent structural patterns in their ASTs

This enables the system to learn new capabilities over time.
"""

from typing import Dict, Any, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import json
import logging
from pathlib import Path

from .trace_analyzer import TraceAnalyzer, TracePattern, create_trace_analyzer

logger = logging.getLogger(__name__)


@dataclass
class EmergentIntent:
    """A newly discovered intent pattern"""
    intent_id: str
    proposed_name: str
    description: str
    frequency: int
    confidence: float

    # AST patterns that characterize this intent
    ast_patterns: List[Dict[str, Any]] = field(default_factory=list)

    # Example queries
    example_queries: List[str] = field(default_factory=list)

    # Why this is considered emergent
    evidence: Dict[str, Any] = field(default_factory=dict)

    # Suggested handling
    suggested_expert: Optional[str] = None
    suggested_implementation: Optional[str] = None


@dataclass
class IntentSignal:
    """A signal that might indicate an emergent intent"""
    signal_type: str  # 'repeated_failure', 'fallback_usage', 'low_confidence', 'similar_asts'
    strength: float  # 0.0 to 1.0
    evidence: Dict[str, Any]


class EmergentIntentAnalyzer:
    """
    Analyzes execution traces to discover new intent patterns.

    Uses a rule-based approach to identify:
    1. Queries that repeatedly fail or use fallback
    2. Queries with consistently low confidence scores
    3. AST patterns that cluster together but don't map to existing intents
    """

    def __init__(self, trace_analyzer: Optional[TraceAnalyzer] = None):
        """
        Initialize Emergent Intent Analyzer.

        Args:
            trace_analyzer: Existing TraceAnalyzer (or creates new one)
        """
        self.trace_analyzer = trace_analyzer or create_trace_analyzer()
        self.emergent_intents: List[EmergentIntent] = []
        self.signals: List[IntentSignal] = []

        # Thresholds for detection
        self.min_frequency = 3  # Minimum occurrences to consider
        self.min_confidence = 0.7  # Minimum confidence for emergent intent

        logger.info("EmergentIntentAnalyzer initialized")

    def analyze(self, trace_directory: Optional[Path] = None) -> List[EmergentIntent]:
        """
        Analyze traces to detect emergent intents.

        Args:
            trace_directory: Directory containing traces

        Returns:
            List of discovered emergent intents
        """
        logger.info("Analyzing for emergent intents")

        # First, run standard trace analysis
        if trace_directory:
            self.trace_analyzer.trace_directory = trace_directory

        stats = self.trace_analyzer.analyze_traces()

        # Detect signals
        self._detect_signals(stats)

        # Cluster signals into emergent intents
        self._cluster_signals_into_intents()

        logger.info(f"Detected {len(self.emergent_intents)} emergent intents "
                   f"from {len(self.signals)} signals")

        return self.emergent_intents

    def _detect_signals(self, stats):
        """Detect signals that might indicate emergent intents"""
        self.signals = []

        # Signal 1: High failure rate on specific error types
        for error_type, count in stats.error_types.items():
            if count >= self.min_frequency:
                signal = IntentSignal(
                    signal_type='repeated_failure',
                    strength=min(count / 10.0, 1.0),  # Scale to 0-1
                    evidence={
                        'error_type': error_type,
                        'count': count,
                        'description': f"{error_type} occurred {count} times"
                    }
                )
                self.signals.append(signal)
                logger.debug(f"Signal detected: {signal.signal_type} - {error_type}")

        # Signal 2: Intents with low success rates
        for expert, success_rate in stats.expert_success_rates.items():
            if success_rate < 0.7:  # Less than 70% success
                count = stats.expert_counts.get(expert, 0)
                if count >= self.min_frequency:
                    signal = IntentSignal(
                        signal_type='low_confidence',
                        strength=1.0 - success_rate,
                        evidence={
                            'expert': expert,
                            'success_rate': success_rate,
                            'count': count,
                            'description': f"{expert} has {success_rate:.1%} success rate"
                        }
                    )
                    self.signals.append(signal)
                    logger.debug(f"Signal detected: {signal.signal_type} - {expert}")

        # Signal 3: Look for patterns in trace patterns
        for pattern in self.trace_analyzer.patterns:
            if pattern.pattern_type == 'error' and pattern.frequency >= self.min_frequency:
                signal = IntentSignal(
                    signal_type='repeated_failure',
                    strength=min(pattern.frequency / 10.0, 1.0),
                    evidence={
                        'pattern': pattern.description,
                        'frequency': pattern.frequency,
                        'metadata': pattern.metadata
                    }
                )
                self.signals.append(signal)

    def _cluster_signals_into_intents(self):
        """Group related signals into emergent intent proposals"""
        self.emergent_intents = []

        # Group signals by type
        signals_by_type = defaultdict(list)
        for signal in self.signals:
            signals_by_type[signal.signal_type].append(signal)

        # Process repeated failures
        if 'repeated_failure' in signals_by_type:
            repeated_failures = signals_by_type['repeated_failure']

            # Group by error type
            error_groups = defaultdict(list)
            for signal in repeated_failures:
                error_type = signal.evidence.get('error_type', 'unknown')
                error_groups[error_type].append(signal)

            # Create emergent intents for each significant error pattern
            for error_type, signals in error_groups.items():
                total_count = sum(s.evidence.get('count', 0) for s in signals)
                avg_strength = sum(s.strength for s in signals) / len(signals)

                if total_count >= self.min_frequency and avg_strength >= 0.3:
                    intent = self._create_emergent_intent_from_error_pattern(
                        error_type, signals, total_count, avg_strength
                    )
                    self.emergent_intents.append(intent)

        # Process low confidence patterns
        if 'low_confidence' in signals_by_type:
            low_conf_signals = signals_by_type['low_confidence']

            for signal in low_conf_signals:
                expert = signal.evidence.get('expert')
                if expert:
                    intent = EmergentIntent(
                        intent_id=f"improve_{expert.lower()}",
                        proposed_name=f"Enhanced_{expert}",
                        description=f"Improved handling for {expert} queries",
                        frequency=signal.evidence.get('count', 0),
                        confidence=signal.strength,
                        evidence=signal.evidence,
                        suggested_expert=expert,
                        suggested_implementation=(
                            f"Review {expert} implementation to improve success rate. "
                            f"Current success rate: {signal.evidence.get('success_rate', 0):.1%}"
                        )
                    )
                    self.emergent_intents.append(intent)

    def _create_emergent_intent_from_error_pattern(
        self,
        error_type: str,
        signals: List[IntentSignal],
        total_count: int,
        avg_strength: float
    ) -> EmergentIntent:
        """Create an emergent intent from an error pattern"""

        # Map error types to suggestions
        suggestions = {
            'parsing_error': {
                'name': 'Extended_Vocabulary_Handler',
                'description': 'Handle queries with unknown Esperanto words',
                'implementation': 'Expand vocabulary or implement fuzzy matching'
            },
            'translation_error': {
                'name': 'Translation_Fallback_Handler',
                'description': 'Handle translation failures gracefully',
                'implementation': 'Add fallback translation services or retry logic'
            },
            'routing_error': {
                'name': 'Enhanced_Router',
                'description': 'Improve expert routing for ambiguous queries',
                'implementation': 'Add more sophisticated intent classification'
            },
            'timeout_error': {
                'name': 'Performance_Optimizer',
                'description': 'Optimize slow operations',
                'implementation': 'Profile and optimize slow code paths'
            }
        }

        suggestion = suggestions.get(error_type, {
            'name': f'Error_Handler_{error_type}',
            'description': f'Handle {error_type} failures',
            'implementation': f'Investigate and fix {error_type} issues'
        })

        return EmergentIntent(
            intent_id=f"emergent_{error_type}_{total_count}",
            proposed_name=suggestion['name'],
            description=suggestion['description'],
            frequency=total_count,
            confidence=avg_strength,
            evidence={
                'error_type': error_type,
                'signals': [s.evidence for s in signals],
                'total_count': total_count
            },
            suggested_implementation=suggestion['implementation']
        )

    def generate_report(self) -> str:
        """Generate human-readable report of emergent intents"""
        lines = []

        lines.append("=" * 80)
        lines.append("EMERGENT INTENT ANALYSIS REPORT")
        lines.append("=" * 80)
        lines.append("")

        lines.append(f"Signals detected: {len(self.signals)}")
        lines.append(f"Emergent intents identified: {len(self.emergent_intents)}")
        lines.append("")

        if not self.emergent_intents:
            lines.append("No emergent intents detected.")
            lines.append("This suggests the system is handling all query types well!")
            lines.append("")
            lines.append("=" * 80)
            return "\n".join(lines)

        lines.append("EMERGENT INTENTS")
        lines.append("-" * 80)

        for i, intent in enumerate(self.emergent_intents, 1):
            lines.append(f"\n{i}. {intent.proposed_name}")
            lines.append(f"   ID: {intent.intent_id}")
            lines.append(f"   Description: {intent.description}")
            lines.append(f"   Frequency: {intent.frequency} occurrences")
            lines.append(f"   Confidence: {intent.confidence:.2f}")

            if intent.suggested_expert:
                lines.append(f"   Suggested Expert: {intent.suggested_expert}")

            if intent.suggested_implementation:
                lines.append(f"   Suggested Implementation:")
                lines.append(f"     {intent.suggested_implementation}")

            if intent.example_queries:
                lines.append(f"   Example Queries:")
                for query in intent.example_queries[:3]:
                    lines.append(f"     - {query}")

        lines.append("")
        lines.append("=" * 80)

        return "\n".join(lines)

    def export_intents(self, output_file: Path):
        """Export emergent intents to JSON file"""
        data = {
            'total_signals': len(self.signals),
            'total_emergent_intents': len(self.emergent_intents),
            'emergent_intents': [
                {
                    'intent_id': intent.intent_id,
                    'proposed_name': intent.proposed_name,
                    'description': intent.description,
                    'frequency': intent.frequency,
                    'confidence': intent.confidence,
                    'example_queries': intent.example_queries,
                    'evidence': intent.evidence,
                    'suggested_expert': intent.suggested_expert,
                    'suggested_implementation': intent.suggested_implementation
                }
                for intent in self.emergent_intents
            ]
        }

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.info(f"Emergent intents exported to {output_file}")

    def propose_improvements(self) -> List[Dict[str, Any]]:
        """
        Generate actionable improvement proposals.

        Returns:
            List of improvement proposals for human review
        """
        proposals = []

        for intent in self.emergent_intents:
            proposal = {
                'title': f"Implement {intent.proposed_name}",
                'description': intent.description,
                'priority': self._calculate_priority(intent),
                'implementation_notes': intent.suggested_implementation,
                'evidence': {
                    'frequency': intent.frequency,
                    'confidence': intent.confidence,
                    'examples': intent.example_queries[:5]
                },
                'estimated_impact': self._estimate_impact(intent)
            }
            proposals.append(proposal)

        # Sort by priority (high to low)
        proposals.sort(key=lambda p: p['priority'], reverse=True)

        return proposals

    def _calculate_priority(self, intent: EmergentIntent) -> float:
        """Calculate priority score for an emergent intent"""
        # Priority based on frequency and confidence
        frequency_score = min(intent.frequency / 20.0, 1.0)
        confidence_score = intent.confidence

        return (frequency_score * 0.6 + confidence_score * 0.4)

    def _estimate_impact(self, intent: EmergentIntent) -> str:
        """Estimate impact of implementing this intent"""
        if intent.frequency >= 10:
            return "HIGH - Affects many queries"
        elif intent.frequency >= 5:
            return "MEDIUM - Affects multiple queries"
        else:
            return "LOW - Affects few queries"


# Factory function
def create_emergent_intent_analyzer(
    trace_analyzer: Optional[TraceAnalyzer] = None
) -> EmergentIntentAnalyzer:
    """
    Create and return an EmergentIntentAnalyzer instance.

    Args:
        trace_analyzer: Existing TraceAnalyzer (or creates new one)

    Returns:
        Initialized EmergentIntentAnalyzer
    """
    return EmergentIntentAnalyzer(trace_analyzer=trace_analyzer)


if __name__ == "__main__":
    # Test emergent intent analyzer
    print("Testing Emergent Intent Analyzer")
    print("=" * 80)

    # Create mock trace directory
    import tempfile
    from .trace_analyzer import create_trace_analyzer

    with tempfile.TemporaryDirectory() as tmpdir:
        trace_dir = Path(tmpdir)

        # Create mock traces with patterns
        mock_traces = [
            # Parsing errors (repeated pattern)
            {
                'trace_id': f'trace_{i:03d}',
                'steps': [
                    {'step': 'Parser', 'status': 'error',
                     'error': 'Ne povis trovi validan radikon en "xyz"'}
                ]
            }
            for i in range(5)
        ] + [
            # Low confidence routing
            {
                'trace_id': f'trace_{i:03d}',
                'steps': [
                    {'step': 'Orchestrator', 'status': 'completed',
                     'result': {'intent': 'general_query', 'expert': 'Fallback'}},
                    {'step': 'Fallback', 'status': 'error', 'error': 'Could not handle'}
                ]
            }
            for i in range(100, 104)
        ]

        # Write traces
        for trace in mock_traces:
            trace_file = trace_dir / f"{trace['trace_id']}.json"
            with open(trace_file, 'w') as f:
                json.dump(trace, f)

        # Analyze
        trace_analyzer = create_trace_analyzer(trace_dir)
        analyzer = create_emergent_intent_analyzer(trace_analyzer)

        intents = analyzer.analyze(trace_dir)

        # Print report
        print(analyzer.generate_report())

        # Get improvement proposals
        proposals = analyzer.propose_improvements()

        print("\nIMPROVEMENT PROPOSALS")
        print("-" * 80)
        for i, proposal in enumerate(proposals, 1):
            print(f"\n{i}. {proposal['title']}")
            print(f"   Priority: {proposal['priority']:.2f}")
            print(f"   Impact: {proposal['estimated_impact']}")
            print(f"   Implementation: {proposal['implementation_notes']}")

        # Export
        export_file = trace_dir / "emergent_intents.json"
        analyzer.export_intents(export_file)
        print(f"\n✅ Emergent intents exported to {export_file}")

    print("\n✅ Emergent Intent Analyzer test complete!")
