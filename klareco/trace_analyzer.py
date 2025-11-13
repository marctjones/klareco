"""
Trace Analysis System - Analyze execution traces for patterns

Part of Phase 9: Learning Loop
Analyzes execution traces to identify:
- Common query patterns
- Performance bottlenecks
- Error patterns
- Successful expert routing patterns

This is the foundation for emergent intent detection and system improvement.
"""

from typing import Dict, Any, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, Counter
import json
import logging
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class TracePattern:
    """A detected pattern in execution traces"""
    pattern_id: str
    pattern_type: str  # 'intent', 'error', 'routing', 'performance'
    description: str
    frequency: int
    examples: List[str] = field(default_factory=list)
    confidence: float = 0.0
    first_seen: Optional[datetime] = None
    last_seen: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TraceStatistics:
    """Aggregated statistics from trace analysis"""
    total_traces: int = 0
    successful_traces: int = 0
    failed_traces: int = 0
    avg_execution_time: float = 0.0

    # Intent distribution
    intent_counts: Dict[str, int] = field(default_factory=dict)

    # Expert routing
    expert_counts: Dict[str, int] = field(default_factory=dict)
    expert_success_rates: Dict[str, float] = field(default_factory=dict)

    # Error patterns
    error_types: Counter = field(default_factory=Counter)

    # Performance metrics
    slow_queries: List[Tuple[str, float]] = field(default_factory=list)

    # Temporal patterns
    queries_by_hour: List[int] = field(default_factory=lambda: [0] * 24)


class TraceAnalyzer:
    """
    Analyzes execution traces to identify patterns and opportunities for improvement.

    This is the foundation of the Learning Loop - understanding what the system
    does well and where it struggles.
    """

    def __init__(self, trace_directory: Optional[Path] = None):
        """
        Initialize Trace Analyzer.

        Args:
            trace_directory: Directory containing execution trace JSON files
        """
        self.trace_directory = trace_directory or Path("execution_traces")
        self.patterns: List[TracePattern] = []
        self.statistics = TraceStatistics()

        logger.info(f"TraceAnalyzer initialized (trace_dir={self.trace_directory})")

    def analyze_traces(self, limit: Optional[int] = None) -> TraceStatistics:
        """
        Analyze all traces in directory.

        Args:
            limit: Maximum number of traces to analyze (None for all)

        Returns:
            Aggregated statistics
        """
        logger.info(f"Analyzing traces in {self.trace_directory}")

        # Reset statistics for fresh analysis
        self.statistics = TraceStatistics()
        self.patterns = []

        traces = self._load_traces(limit)

        if not traces:
            logger.warning("No traces found to analyze")
            return self.statistics

        # Analyze each trace
        for trace in traces:
            self._analyze_single_trace(trace)

        # Compute derived statistics
        self._compute_statistics()

        # Detect patterns
        self._detect_patterns()

        logger.info(f"Analyzed {self.statistics.total_traces} traces, "
                   f"found {len(self.patterns)} patterns")

        return self.statistics

    def _load_traces(self, limit: Optional[int]) -> List[Dict[str, Any]]:
        """Load trace files from directory"""
        traces = []

        if not self.trace_directory.exists():
            logger.warning(f"Trace directory {self.trace_directory} does not exist")
            return traces

        # Find all JSON files
        trace_files = sorted(self.trace_directory.glob("*.json"))

        if limit:
            trace_files = trace_files[:limit]

        for trace_file in trace_files:
            try:
                with open(trace_file, 'r', encoding='utf-8') as f:
                    trace = json.load(f)
                    traces.append(trace)
            except Exception as e:
                logger.error(f"Failed to load {trace_file}: {e}")

        return traces

    def _analyze_single_trace(self, trace: Dict[str, Any]):
        """Analyze a single execution trace"""
        self.statistics.total_traces += 1

        # Extract key information
        trace_id = trace.get('trace_id', 'unknown')
        steps = trace.get('steps', [])

        if not steps:
            return

        # Check success/failure
        final_step = steps[-1]
        if final_step.get('status') == 'completed':
            self.statistics.successful_traces += 1
        else:
            self.statistics.failed_traces += 1

        # Extract intent if available
        for step in steps:
            if step.get('step') == 'Orchestrator':
                result = step.get('result', {})
                intent = result.get('intent')
                if intent:
                    self.statistics.intent_counts[intent] = \
                        self.statistics.intent_counts.get(intent, 0) + 1

                expert = result.get('expert')
                if expert:
                    self.statistics.expert_counts[expert] = \
                        self.statistics.expert_counts.get(expert, 0) + 1

        # Extract error information
        for step in steps:
            if step.get('status') == 'error':
                error = step.get('error', 'unknown_error')
                # Extract error type from message
                error_type = self._classify_error(error)
                self.statistics.error_types[error_type] += 1

    def _classify_error(self, error_msg: str) -> str:
        """Classify error type from error message"""
        error_lower = error_msg.lower()

        if 'parse' in error_lower or 'radiko' in error_lower:
            return 'parsing_error'
        elif 'timeout' in error_lower:
            return 'timeout_error'
        elif 'translation' in error_lower or 'translat' in error_lower:
            return 'translation_error'
        elif 'expert' in error_lower or 'rout' in error_lower:
            return 'routing_error'
        else:
            return 'other_error'

    def _compute_statistics(self):
        """Compute derived statistics"""
        # Success rate per expert
        for expert, count in self.statistics.expert_counts.items():
            # This is simplified - would need success/failure per expert
            # For now, use overall success rate
            if self.statistics.total_traces > 0:
                overall_rate = self.statistics.successful_traces / self.statistics.total_traces
                self.statistics.expert_success_rates[expert] = overall_rate

    def _detect_patterns(self):
        """Detect patterns in the analyzed data"""
        self.patterns = []

        # Pattern 1: High-frequency intents
        for intent, count in self.statistics.intent_counts.items():
            if count >= 5:  # Threshold for "common"
                pattern = TracePattern(
                    pattern_id=f"intent_{intent}_{count}",
                    pattern_type="intent",
                    description=f"Common intent: {intent}",
                    frequency=count,
                    confidence=0.9,
                    metadata={'intent': intent}
                )
                self.patterns.append(pattern)

        # Pattern 2: Frequent error types
        for error_type, count in self.statistics.error_types.most_common(5):
            if count >= 3:
                pattern = TracePattern(
                    pattern_id=f"error_{error_type}_{count}",
                    pattern_type="error",
                    description=f"Recurring error: {error_type}",
                    frequency=count,
                    confidence=0.8,
                    metadata={'error_type': error_type}
                )
                self.patterns.append(pattern)

        # Pattern 3: Popular experts
        for expert, count in self.statistics.expert_counts.items():
            if count >= 5:
                pattern = TracePattern(
                    pattern_id=f"expert_{expert}_{count}",
                    pattern_type="routing",
                    description=f"Frequently used expert: {expert}",
                    frequency=count,
                    confidence=0.9,
                    metadata={'expert': expert}
                )
                self.patterns.append(pattern)

    def get_patterns_by_type(self, pattern_type: str) -> List[TracePattern]:
        """Get all patterns of a specific type"""
        return [p for p in self.patterns if p.pattern_type == pattern_type]

    def get_top_patterns(self, n: int = 10) -> List[TracePattern]:
        """Get top N patterns by frequency"""
        return sorted(self.patterns, key=lambda p: p.frequency, reverse=True)[:n]

    def generate_report(self) -> str:
        """Generate human-readable analysis report"""
        lines = []

        lines.append("=" * 80)
        lines.append("EXECUTION TRACE ANALYSIS REPORT")
        lines.append("=" * 80)
        lines.append("")

        # Summary statistics
        lines.append("SUMMARY STATISTICS")
        lines.append("-" * 80)
        lines.append(f"Total traces analyzed: {self.statistics.total_traces}")
        lines.append(f"Successful: {self.statistics.successful_traces}")
        lines.append(f"Failed: {self.statistics.failed_traces}")

        if self.statistics.total_traces > 0:
            success_rate = (self.statistics.successful_traces /
                          self.statistics.total_traces * 100)
            lines.append(f"Success rate: {success_rate:.1f}%")
        lines.append("")

        # Intent distribution
        if self.statistics.intent_counts:
            lines.append("INTENT DISTRIBUTION")
            lines.append("-" * 80)
            for intent, count in sorted(self.statistics.intent_counts.items(),
                                       key=lambda x: x[1], reverse=True):
                lines.append(f"  {intent}: {count}")
            lines.append("")

        # Expert usage
        if self.statistics.expert_counts:
            lines.append("EXPERT USAGE")
            lines.append("-" * 80)
            for expert, count in sorted(self.statistics.expert_counts.items(),
                                       key=lambda x: x[1], reverse=True):
                success_rate = self.statistics.expert_success_rates.get(expert, 0) * 100
                lines.append(f"  {expert}: {count} calls ({success_rate:.1f}% success)")
            lines.append("")

        # Error patterns
        if self.statistics.error_types:
            lines.append("ERROR PATTERNS")
            lines.append("-" * 80)
            for error_type, count in self.statistics.error_types.most_common(10):
                lines.append(f"  {error_type}: {count}")
            lines.append("")

        # Detected patterns
        if self.patterns:
            lines.append("DETECTED PATTERNS")
            lines.append("-" * 80)
            for pattern in self.get_top_patterns(10):
                lines.append(f"  [{pattern.pattern_type.upper()}] {pattern.description}")
                lines.append(f"    Frequency: {pattern.frequency}, "
                           f"Confidence: {pattern.confidence:.2f}")
            lines.append("")

        lines.append("=" * 80)

        return "\n".join(lines)

    def export_statistics(self, output_file: Path):
        """Export statistics to JSON file"""
        data = {
            'total_traces': self.statistics.total_traces,
            'successful_traces': self.statistics.successful_traces,
            'failed_traces': self.statistics.failed_traces,
            'intent_counts': self.statistics.intent_counts,
            'expert_counts': self.statistics.expert_counts,
            'expert_success_rates': self.statistics.expert_success_rates,
            'error_types': dict(self.statistics.error_types),
            'patterns': [
                {
                    'pattern_id': p.pattern_id,
                    'pattern_type': p.pattern_type,
                    'description': p.description,
                    'frequency': p.frequency,
                    'confidence': p.confidence,
                    'metadata': p.metadata
                }
                for p in self.patterns
            ]
        }

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.info(f"Statistics exported to {output_file}")


# Factory function
def create_trace_analyzer(trace_directory: Optional[Path] = None) -> TraceAnalyzer:
    """
    Create and return a TraceAnalyzer instance.

    Args:
        trace_directory: Directory containing trace files

    Returns:
        Initialized TraceAnalyzer
    """
    return TraceAnalyzer(trace_directory=trace_directory)


if __name__ == "__main__":
    # Test trace analyzer with mock data
    print("Testing Trace Analyzer")
    print("=" * 80)

    # Create mock trace directory
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        trace_dir = Path(tmpdir)

        # Create mock traces
        mock_traces = [
            {
                'trace_id': 'trace_001',
                'steps': [
                    {'step': 'Orchestrator', 'status': 'completed',
                     'result': {'intent': 'calculation_request', 'expert': 'MathExpert'}},
                    {'step': 'MathExpert', 'status': 'completed'}
                ]
            },
            {
                'trace_id': 'trace_002',
                'steps': [
                    {'step': 'Orchestrator', 'status': 'completed',
                     'result': {'intent': 'temporal_query', 'expert': 'DateExpert'}},
                    {'step': 'DateExpert', 'status': 'completed'}
                ]
            },
            {
                'trace_id': 'trace_003',
                'steps': [
                    {'step': 'Parser', 'status': 'error',
                     'error': 'Ne povis trovi validan radikon'}
                ]
            }
        ]

        # Write mock traces
        for trace in mock_traces:
            trace_file = trace_dir / f"{trace['trace_id']}.json"
            with open(trace_file, 'w') as f:
                json.dump(trace, f)

        # Analyze traces
        analyzer = create_trace_analyzer(trace_dir)
        stats = analyzer.analyze_traces()

        # Print report
        print(analyzer.generate_report())

        # Export statistics
        export_file = trace_dir / "statistics.json"
        analyzer.export_statistics(export_file)
        print(f"\n✅ Statistics exported to {export_file}")

    print("\n✅ Trace Analyzer test complete!")
