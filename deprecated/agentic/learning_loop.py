"""
Learning Loop - Complete system for continuous improvement

Part of Phase 9: Learning Loop
Orchestrates the complete learning cycle:
1. Analyze execution traces
2. Detect emergent intents
3. Generate PR proposals
4. Human review and approval

This is the top-level interface for the self-improvement system.
"""

from typing import Dict, Any, List, Optional
from pathlib import Path
import logging
from datetime import datetime

from .trace_analyzer import TraceAnalyzer, create_trace_analyzer
from .emergent_intent_analyzer import EmergentIntentAnalyzer, create_emergent_intent_analyzer
from .pr_generator import PRGenerator, create_pr_generator, PRProposal

logger = logging.getLogger(__name__)


class LearningLoop:
    """
    Complete learning loop implementation.

    Orchestrates the full cycle from trace analysis to PR generation,
    enabling continuous system improvement through human-in-the-loop governance.
    """

    def __init__(
        self,
        trace_directory: Optional[Path] = None,
        output_directory: Optional[Path] = None,
        repo_path: Optional[Path] = None
    ):
        """
        Initialize Learning Loop.

        Args:
            trace_directory: Directory containing execution traces
            output_directory: Directory for analysis outputs
            repo_path: Path to git repository
        """
        self.trace_directory = trace_directory or Path("execution_traces")
        self.output_directory = output_directory or Path("learning_loop_output")
        self.repo_path = repo_path or Path.cwd()

        # Create output directory
        self.output_directory.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.trace_analyzer = create_trace_analyzer(self.trace_directory)
        self.intent_analyzer = create_emergent_intent_analyzer(self.trace_analyzer)
        self.pr_generator = create_pr_generator(self.repo_path)

        # Results
        self.latest_run: Optional[Dict[str, Any]] = None

        logger.info(f"LearningLoop initialized")
        logger.info(f"  Trace directory: {self.trace_directory}")
        logger.info(f"  Output directory: {self.output_directory}")

    def run_full_cycle(
        self,
        max_proposals: int = 5,
        min_intent_frequency: int = 3
    ) -> Dict[str, Any]:
        """
        Run complete learning cycle.

        Args:
            max_proposals: Maximum number of PR proposals to generate
            min_intent_frequency: Minimum frequency for emergent intent detection

        Returns:
            Summary of learning cycle results
        """
        logger.info("=" * 80)
        logger.info("STARTING LEARNING LOOP CYCLE")
        logger.info("=" * 80)

        start_time = datetime.now()

        # Step 1: Analyze traces
        logger.info("\nStep 1: Analyzing execution traces...")
        trace_stats = self.trace_analyzer.analyze_traces()

        logger.info(f"  Analyzed {trace_stats.total_traces} traces")
        logger.info(f"  Success rate: {self._calculate_success_rate(trace_stats):.1%}")

        # Step 2: Detect emergent intents
        logger.info("\nStep 2: Detecting emergent intents...")
        self.intent_analyzer.min_frequency = min_intent_frequency
        emergent_intents = self.intent_analyzer.analyze(self.trace_directory)

        logger.info(f"  Detected {len(emergent_intents)} emergent intents")

        # Step 3: Generate PR proposals
        logger.info("\nStep 3: Generating PR proposals...")
        if emergent_intents:
            proposals = self.pr_generator.generate_pr_batch(
                emergent_intents,
                max_prs=max_proposals
            )
            logger.info(f"  Generated {len(proposals)} PR proposals")
        else:
            proposals = []
            logger.info("  No proposals generated (no emergent intents)")

        # Step 4: Export results
        logger.info("\nStep 4: Exporting results...")
        self._export_results(trace_stats, emergent_intents, proposals)

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        # Build summary
        summary = {
            'start_time': start_time.isoformat(),
            'end_time': end_time.isoformat(),
            'duration_seconds': duration,
            'traces_analyzed': trace_stats.total_traces,
            'success_rate': self._calculate_success_rate(trace_stats),
            'emergent_intents_found': len(emergent_intents),
            'pr_proposals_generated': len(proposals),
            'output_directory': str(self.output_directory)
        }

        self.latest_run = summary

        logger.info("\n" + "=" * 80)
        logger.info("LEARNING LOOP CYCLE COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Duration: {duration:.2f}s")
        logger.info(f"Results saved to: {self.output_directory}")

        return summary

    def _calculate_success_rate(self, trace_stats) -> float:
        """Calculate success rate from trace statistics"""
        if trace_stats.total_traces == 0:
            return 0.0
        return trace_stats.successful_traces / trace_stats.total_traces

    def _export_results(self, trace_stats, emergent_intents, proposals):
        """Export all results to output directory"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Export trace analysis
        trace_report_file = self.output_directory / f"trace_analysis_{timestamp}.txt"
        with open(trace_report_file, 'w', encoding='utf-8') as f:
            f.write(self.trace_analyzer.generate_report())
        logger.info(f"  Trace analysis: {trace_report_file}")

        trace_stats_file = self.output_directory / f"trace_statistics_{timestamp}.json"
        self.trace_analyzer.export_statistics(trace_stats_file)
        logger.info(f"  Trace statistics: {trace_stats_file}")

        # Export emergent intents
        if emergent_intents:
            intent_report_file = self.output_directory / f"emergent_intents_{timestamp}.txt"
            with open(intent_report_file, 'w', encoding='utf-8') as f:
                f.write(self.intent_analyzer.generate_report())
            logger.info(f"  Emergent intents: {intent_report_file}")

            intent_json_file = self.output_directory / f"emergent_intents_{timestamp}.json"
            self.intent_analyzer.export_intents(intent_json_file)
            logger.info(f"  Intent data: {intent_json_file}")

        # Export PR proposals
        if proposals:
            pr_report_file = self.output_directory / f"pr_proposals_{timestamp}.txt"
            with open(pr_report_file, 'w', encoding='utf-8') as f:
                f.write(self.pr_generator.generate_report())
            logger.info(f"  PR proposals: {pr_report_file}")

            pr_json_file = self.output_directory / f"pr_proposals_{timestamp}.json"
            self.pr_generator.export_proposals(pr_json_file)
            logger.info(f"  PR data: {pr_json_file}")

            # Write individual PR markdown files
            pr_markdown_dir = self.output_directory / f"pr_markdown_{timestamp}"
            for proposal in proposals:
                self.pr_generator.write_pr_markdown(proposal, pr_markdown_dir)
            logger.info(f"  PR markdown files: {pr_markdown_dir}")

    def generate_comprehensive_report(self) -> str:
        """Generate comprehensive report of latest run"""
        if not self.latest_run:
            return "No learning loop runs completed yet."

        lines = []

        lines.append("=" * 80)
        lines.append("LEARNING LOOP COMPREHENSIVE REPORT")
        lines.append("=" * 80)
        lines.append("")

        # Summary
        lines.append("SUMMARY")
        lines.append("-" * 80)
        lines.append(f"Start time: {self.latest_run['start_time']}")
        lines.append(f"Duration: {self.latest_run['duration_seconds']:.2f}s")
        lines.append(f"Traces analyzed: {self.latest_run['traces_analyzed']}")
        lines.append(f"Success rate: {self.latest_run['success_rate']:.1%}")
        lines.append(f"Emergent intents: {self.latest_run['emergent_intents_found']}")
        lines.append(f"PR proposals: {self.latest_run['pr_proposals_generated']}")
        lines.append("")

        # Trace analysis
        lines.append(self.trace_analyzer.generate_report())
        lines.append("")

        # Emergent intents
        lines.append(self.intent_analyzer.generate_report())
        lines.append("")

        # PR proposals
        lines.append(self.pr_generator.generate_report())
        lines.append("")

        lines.append("=" * 80)

        return "\n".join(lines)

    def get_actionable_improvements(self) -> List[Dict[str, Any]]:
        """
        Get list of actionable improvements for human review.

        Returns:
            List of improvement proposals with priority and implementation details
        """
        return self.intent_analyzer.propose_improvements()


# Factory function
def create_learning_loop(
    trace_directory: Optional[Path] = None,
    output_directory: Optional[Path] = None,
    repo_path: Optional[Path] = None
) -> LearningLoop:
    """
    Create and return a LearningLoop instance.

    Args:
        trace_directory: Directory containing traces
        output_directory: Directory for outputs
        repo_path: Path to repository

    Returns:
        Initialized LearningLoop
    """
    return LearningLoop(
        trace_directory=trace_directory,
        output_directory=output_directory,
        repo_path=repo_path
    )


if __name__ == "__main__":
    # Test learning loop
    print("Testing Learning Loop")
    print("=" * 80)

    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create mock trace directory
        trace_dir = tmpdir / "traces"
        trace_dir.mkdir()

        # Create mock traces
        import json

        mock_traces = [
            # Successful traces
            {
                'trace_id': f'trace_{i:03d}',
                'steps': [
                    {'step': 'Orchestrator', 'status': 'completed',
                     'result': {'intent': 'calculation_request', 'expert': 'MathExpert'}},
                    {'step': 'MathExpert', 'status': 'completed'}
                ]
            }
            for i in range(10)
        ] + [
            # Failed traces (parsing errors)
            {
                'trace_id': f'trace_{i:03d}',
                'steps': [
                    {'step': 'Parser', 'status': 'error',
                     'error': 'Ne povis trovi validan radikon'}
                ]
            }
            for i in range(100, 105)
        ]

        for trace in mock_traces:
            trace_file = trace_dir / f"{trace['trace_id']}.json"
            with open(trace_file, 'w') as f:
                json.dump(trace, f)

        # Run learning loop
        output_dir = tmpdir / "output"
        learning_loop = create_learning_loop(
            trace_directory=trace_dir,
            output_directory=output_dir,
            repo_path=tmpdir
        )

        summary = learning_loop.run_full_cycle(
            max_proposals=3,
            min_intent_frequency=3
        )

        print("\n" + "=" * 80)
        print("LEARNING LOOP SUMMARY")
        print("=" * 80)
        print(f"Traces analyzed: {summary['traces_analyzed']}")
        print(f"Success rate: {summary['success_rate']:.1%}")
        print(f"Emergent intents: {summary['emergent_intents_found']}")
        print(f"PR proposals: {summary['pr_proposals_generated']}")
        print(f"Duration: {summary['duration_seconds']:.2f}s")

        # Get actionable improvements
        improvements = learning_loop.get_actionable_improvements()
        print(f"\nActionable improvements: {len(improvements)}")
        for imp in improvements:
            print(f"  - {imp['title']} (Priority: {imp['priority']:.2f})")

        print(f"\n✅ Results saved to: {output_dir}")

    print("\n✅ Learning Loop test complete!")
