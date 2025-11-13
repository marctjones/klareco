"""
Unit tests for Learning Loop (Phase 9)
"""

import pytest
import json
import tempfile
from pathlib import Path
from datetime import datetime

from klareco.trace_analyzer import (
    TraceAnalyzer,
    TracePattern,
    TraceStatistics,
    create_trace_analyzer
)
from klareco.emergent_intent_analyzer import (
    EmergentIntentAnalyzer,
    EmergentIntent,
    IntentSignal,
    create_emergent_intent_analyzer
)
from klareco.pr_generator import (
    PRGenerator,
    PRProposal,
    create_pr_generator
)
from klareco.learning_loop import (
    LearningLoop,
    create_learning_loop
)


class TestTraceAnalyzer:
    """Test Trace Analyzer"""

    def test_create_trace_analyzer(self):
        """Test creating trace analyzer"""
        analyzer = create_trace_analyzer()

        assert analyzer is not None
        assert analyzer.trace_directory == Path("execution_traces")
        assert analyzer.statistics.total_traces == 0

    def test_analyze_empty_directory(self, tmp_path):
        """Test analyzing empty directory"""
        analyzer = create_trace_analyzer(tmp_path)
        stats = analyzer.analyze_traces()

        assert stats.total_traces == 0
        assert stats.successful_traces == 0
        assert stats.failed_traces == 0

    def test_analyze_single_trace(self, tmp_path):
        """Test analyzing a single trace"""
        # Create mock trace
        trace = {
            'trace_id': 'test_001',
            'steps': [
                {
                    'step': 'Orchestrator',
                    'status': 'completed',
                    'result': {
                        'intent': 'calculation_request',
                        'expert': 'MathExpert'
                    }
                },
                {'step': 'MathExpert', 'status': 'completed'}
            ]
        }

        trace_file = tmp_path / "test_001.json"
        with open(trace_file, 'w') as f:
            json.dump(trace, f)

        # Analyze
        analyzer = create_trace_analyzer(tmp_path)
        stats = analyzer.analyze_traces()

        assert stats.total_traces == 1
        assert stats.successful_traces == 1
        assert 'calculation_request' in stats.intent_counts
        assert 'MathExpert' in stats.expert_counts

    def test_analyze_failed_trace(self, tmp_path):
        """Test analyzing a failed trace"""
        trace = {
            'trace_id': 'test_002',
            'steps': [
                {
                    'step': 'Parser',
                    'status': 'error',
                    'error': 'Ne povis trovi validan radikon'
                }
            ]
        }

        trace_file = tmp_path / "test_002.json"
        with open(trace_file, 'w') as f:
            json.dump(trace, f)

        analyzer = create_trace_analyzer(tmp_path)
        stats = analyzer.analyze_traces()

        assert stats.total_traces == 1
        assert stats.failed_traces == 1
        assert 'parsing_error' in stats.error_types

    def test_error_classification(self, tmp_path):
        """Test error type classification"""
        analyzer = create_trace_analyzer(tmp_path)

        assert analyzer._classify_error("Ne povis parse") == "parsing_error"
        assert analyzer._classify_error("timeout exceeded") == "timeout_error"
        assert analyzer._classify_error("translation failed") == "translation_error"
        assert analyzer._classify_error("routing error") == "routing_error"
        assert analyzer._classify_error("unknown error") == "other_error"

    def test_pattern_detection(self, tmp_path):
        """Test pattern detection"""
        # Create multiple traces with same intent
        for i in range(5):
            trace = {
                'trace_id': f'test_{i:03d}',
                'steps': [
                    {
                        'step': 'Orchestrator',
                        'status': 'completed',
                        'result': {'intent': 'temporal_query', 'expert': 'DateExpert'}
                    }
                ]
            }
            trace_file = tmp_path / f"test_{i:03d}.json"
            with open(trace_file, 'w') as f:
                json.dump(trace, f)

        analyzer = create_trace_analyzer(tmp_path)
        stats = analyzer.analyze_traces()

        # Should detect pattern for common intent
        intent_patterns = analyzer.get_patterns_by_type('intent')
        assert len(intent_patterns) > 0
        assert any('temporal_query' in p.metadata.get('intent', '') for p in intent_patterns)

    def test_generate_report(self, tmp_path):
        """Test report generation"""
        trace = {
            'trace_id': 'test_001',
            'steps': [
                {'step': 'Orchestrator', 'status': 'completed',
                 'result': {'intent': 'calculation_request', 'expert': 'MathExpert'}},
                {'step': 'MathExpert', 'status': 'completed'}
            ]
        }

        trace_file = tmp_path / "test_001.json"
        with open(trace_file, 'w') as f:
            json.dump(trace, f)

        analyzer = create_trace_analyzer(tmp_path)
        analyzer.analyze_traces()

        report = analyzer.generate_report()

        assert "EXECUTION TRACE ANALYSIS REPORT" in report
        assert "Total traces analyzed: 1" in report
        assert "INTENT DISTRIBUTION" in report

    def test_export_statistics(self, tmp_path):
        """Test statistics export"""
        trace = {
            'trace_id': 'test_001',
            'steps': [{'step': 'Test', 'status': 'completed'}]
        }

        trace_file = tmp_path / "test_001.json"
        with open(trace_file, 'w') as f:
            json.dump(trace, f)

        analyzer = create_trace_analyzer(tmp_path)
        analyzer.analyze_traces()

        export_file = tmp_path / "stats.json"
        analyzer.export_statistics(export_file)

        assert export_file.exists()

        with open(export_file) as f:
            data = json.load(f)

        assert data['total_traces'] == 1


class TestEmergentIntentAnalyzer:
    """Test Emergent Intent Analyzer"""

    def test_create_analyzer(self):
        """Test creating analyzer"""
        analyzer = create_emergent_intent_analyzer()

        assert analyzer is not None
        assert analyzer.min_frequency == 3
        assert analyzer.min_confidence == 0.7

    def test_detect_repeated_failures(self, tmp_path):
        """Test detection of repeated failures"""
        # Create traces with repeated parsing errors
        for i in range(5):
            trace = {
                'trace_id': f'test_{i:03d}',
                'steps': [
                    {'step': 'Parser', 'status': 'error',
                     'error': 'Ne povis trovi validan radikon'}
                ]
            }
            trace_file = tmp_path / f"test_{i:03d}.json"
            with open(trace_file, 'w') as f:
                json.dump(trace, f)

        analyzer = create_emergent_intent_analyzer()
        intents = analyzer.analyze(tmp_path)

        # Should detect emergent intent from repeated failures
        assert len(intents) > 0
        assert any('parsing_error' in intent.evidence.get('error_type', '')
                  for intent in intents)

    def test_detect_low_confidence(self, tmp_path):
        """Test detection of low confidence patterns"""
        # Create traces with low success expert
        for i in range(5):
            trace = {
                'trace_id': f'test_{i:03d}',
                'steps': [
                    {'step': 'Orchestrator', 'status': 'completed',
                     'result': {'intent': 'test', 'expert': 'TestExpert'}},
                    {'step': 'TestExpert', 'status': 'error', 'error': 'Failed'}
                ]
            }
            trace_file = tmp_path / f"test_{i:03d}.json"
            with open(trace_file, 'w') as f:
                json.dump(trace, f)

        analyzer = create_emergent_intent_analyzer()
        intents = analyzer.analyze(tmp_path)

        # Should detect issues with TestExpert
        assert len(intents) > 0

    def test_signal_clustering(self, tmp_path):
        """Test signal clustering into intents"""
        # Create mixed error patterns
        errors = [
            ('parsing_error', 'Ne povis parse'),
            ('parsing_error', 'Invalid radiko'),
            ('parsing_error', 'Parse failed')
        ]

        for i, (error_type, error_msg) in enumerate(errors):
            trace = {
                'trace_id': f'test_{i:03d}',
                'steps': [
                    {'step': 'Parser', 'status': 'error', 'error': error_msg}
                ]
            }
            trace_file = tmp_path / f"test_{i:03d}.json"
            with open(trace_file, 'w') as f:
                json.dump(trace, f)

        analyzer = create_emergent_intent_analyzer()
        intents = analyzer.analyze(tmp_path)

        # Should cluster into single parsing intent
        parsing_intents = [i for i in intents
                          if 'parsing' in i.evidence.get('error_type', '')]
        assert len(parsing_intents) > 0

    def test_generate_report(self, tmp_path):
        """Test report generation"""
        analyzer = create_emergent_intent_analyzer()
        intents = analyzer.analyze(tmp_path)

        report = analyzer.generate_report()

        assert "EMERGENT INTENT ANALYSIS REPORT" in report

    def test_propose_improvements(self, tmp_path):
        """Test improvement proposals"""
        # Create traces with errors
        for i in range(5):
            trace = {
                'trace_id': f'test_{i:03d}',
                'steps': [{'step': 'Parser', 'status': 'error', 'error': 'Failed'}]
            }
            trace_file = tmp_path / f"test_{i:03d}.json"
            with open(trace_file, 'w') as f:
                json.dump(trace, f)

        analyzer = create_emergent_intent_analyzer()
        analyzer.analyze(tmp_path)

        proposals = analyzer.propose_improvements()

        assert isinstance(proposals, list)
        for proposal in proposals:
            assert 'title' in proposal
            assert 'priority' in proposal
            assert 'estimated_impact' in proposal


class TestPRGenerator:
    """Test PR Generator"""

    def test_create_generator(self):
        """Test creating PR generator"""
        generator = create_pr_generator()

        assert generator is not None
        assert generator.repo_path == Path.cwd()

    def test_generate_pr_from_intent(self):
        """Test PR generation from intent"""
        intent = EmergentIntent(
            intent_id="test_001",
            proposed_name="Test_Expert",
            description="Test expert for testing",
            frequency=5,
            confidence=0.8,
            suggested_implementation="Implement test expert"
        )

        generator = create_pr_generator()
        proposal = generator.generate_pr_from_intent(intent)

        assert proposal is not None
        assert proposal.title == "Implement Test_Expert"
        assert proposal.priority > 0
        assert proposal.status == "proposed"

    def test_pr_description_format(self):
        """Test PR description formatting"""
        intent = EmergentIntent(
            intent_id="test_001",
            proposed_name="Enhanced_Parser",
            description="Improve parser accuracy",
            frequency=10,
            confidence=0.9,
            evidence={'error_type': 'parsing_error'},
            suggested_implementation="Add fuzzy matching"
        )

        generator = create_pr_generator()
        proposal = generator.generate_pr_from_intent(intent)

        # Check description format
        assert "## Enhanced_Parser" in proposal.description
        assert "Frequency:" in proposal.description
        assert "Confidence:" in proposal.description
        assert "Testing" in proposal.description

    def test_suggest_files_to_create(self):
        """Test file creation suggestions"""
        intent = EmergentIntent(
            intent_id="test_001",
            proposed_name="New_Expert",
            description="New expert",
            frequency=5,
            confidence=0.8
        )

        generator = create_pr_generator()
        proposal = generator.generate_pr_from_intent(intent)

        # Should suggest creating expert file
        assert len(proposal.files_to_create) > 0
        assert any('expert' in f['path'] for f in proposal.files_to_create)

    def test_suggest_files_to_modify(self):
        """Test file modification suggestions"""
        intent = EmergentIntent(
            intent_id="test_001",
            proposed_name="Parser_Fix",
            description="Fix parser",
            frequency=5,
            confidence=0.8,
            evidence={'error_type': 'parsing_error'}
        )

        generator = create_pr_generator()
        proposal = generator.generate_pr_from_intent(intent)

        # Should suggest modifying parser
        assert len(proposal.files_to_modify) > 0
        assert any('parser.py' in f['path'] for f in proposal.files_to_modify)

    def test_generate_pr_batch(self):
        """Test batch PR generation"""
        intents = [
            EmergentIntent(
                intent_id=f"test_{i:03d}",
                proposed_name=f"Intent_{i}",
                description=f"Intent {i}",
                frequency=10 - i,  # Decreasing frequency
                confidence=0.8
            )
            for i in range(5)
        ]

        generator = create_pr_generator()
        proposals = generator.generate_pr_batch(intents, max_prs=3)

        # Should return top 3 by priority
        assert len(proposals) == 3
        # Should be sorted by priority (which depends on frequency)
        assert proposals[0].priority >= proposals[1].priority

    def test_generate_report(self):
        """Test report generation"""
        intent = EmergentIntent(
            intent_id="test_001",
            proposed_name="Test",
            description="Test",
            frequency=5,
            confidence=0.8
        )

        generator = create_pr_generator()
        generator.generate_pr_from_intent(intent)

        report = generator.generate_report()

        assert "PULL REQUEST PROPOSALS" in report
        assert "Test" in report

    def test_export_proposals(self, tmp_path):
        """Test proposal export"""
        intent = EmergentIntent(
            intent_id="test_001",
            proposed_name="Test",
            description="Test",
            frequency=5,
            confidence=0.8
        )

        generator = create_pr_generator()
        generator.generate_pr_from_intent(intent)

        export_file = tmp_path / "proposals.json"
        generator.export_proposals(export_file)

        assert export_file.exists()

        with open(export_file) as f:
            data = json.load(f)

        assert data['total_proposals'] == 1

    def test_write_pr_markdown(self, tmp_path):
        """Test PR markdown writing"""
        intent = EmergentIntent(
            intent_id="test_001",
            proposed_name="Test",
            description="Test",
            frequency=5,
            confidence=0.8
        )

        generator = create_pr_generator()
        proposal = generator.generate_pr_from_intent(intent)

        markdown_file = generator.write_pr_markdown(proposal, tmp_path)

        assert markdown_file.exists()
        assert markdown_file.read_text()


class TestLearningLoop:
    """Test Learning Loop"""

    def test_create_learning_loop(self):
        """Test creating learning loop"""
        loop = create_learning_loop()

        assert loop is not None
        assert loop.trace_analyzer is not None
        assert loop.intent_analyzer is not None
        assert loop.pr_generator is not None

    def test_run_full_cycle_no_traces(self, tmp_path):
        """Test running cycle with no traces"""
        trace_dir = tmp_path / "traces"
        trace_dir.mkdir()

        output_dir = tmp_path / "output"

        loop = create_learning_loop(
            trace_directory=trace_dir,
            output_directory=output_dir
        )

        summary = loop.run_full_cycle()

        assert summary['traces_analyzed'] == 0
        assert summary['emergent_intents_found'] == 0
        assert summary['pr_proposals_generated'] == 0

    def test_run_full_cycle_with_traces(self, tmp_path):
        """Test running cycle with traces"""
        trace_dir = tmp_path / "traces"
        trace_dir.mkdir()

        # Create mock traces
        for i in range(5):
            trace = {
                'trace_id': f'test_{i:03d}',
                'steps': [
                    {'step': 'Parser', 'status': 'error',
                     'error': 'Parsing error'}
                ]
            }
            trace_file = trace_dir / f"test_{i:03d}.json"
            with open(trace_file, 'w') as f:
                json.dump(trace, f)

        output_dir = tmp_path / "output"

        loop = create_learning_loop(
            trace_directory=trace_dir,
            output_directory=output_dir
        )

        summary = loop.run_full_cycle(min_intent_frequency=3)

        assert summary['traces_analyzed'] == 5
        assert output_dir.exists()

    def test_generate_comprehensive_report(self, tmp_path):
        """Test comprehensive report generation"""
        trace_dir = tmp_path / "traces"
        trace_dir.mkdir()

        trace = {
            'trace_id': 'test_001',
            'steps': [{'step': 'Test', 'status': 'completed'}]
        }
        trace_file = trace_dir / "test_001.json"
        with open(trace_file, 'w') as f:
            json.dump(trace, f)

        output_dir = tmp_path / "output"

        loop = create_learning_loop(
            trace_directory=trace_dir,
            output_directory=output_dir
        )

        loop.run_full_cycle()

        report = loop.generate_comprehensive_report()

        assert "LEARNING LOOP COMPREHENSIVE REPORT" in report

    def test_get_actionable_improvements(self, tmp_path):
        """Test getting actionable improvements"""
        trace_dir = tmp_path / "traces"
        trace_dir.mkdir()

        # Create traces with errors
        for i in range(5):
            trace = {
                'trace_id': f'test_{i:03d}',
                'steps': [{'step': 'Parser', 'status': 'error', 'error': 'Failed'}]
            }
            trace_file = trace_dir / f"test_{i:03d}.json"
            with open(trace_file, 'w') as f:
                json.dump(trace, f)

        output_dir = tmp_path / "output"

        loop = create_learning_loop(
            trace_directory=trace_dir,
            output_directory=output_dir
        )

        loop.run_full_cycle(min_intent_frequency=3)

        improvements = loop.get_actionable_improvements()

        assert isinstance(improvements, list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
