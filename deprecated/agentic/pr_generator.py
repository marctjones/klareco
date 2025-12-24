"""
Pull Request Generator - Generate PRs for human review of system improvements

Part of Phase 9: Learning Loop
Generates GitHub pull request proposals based on emergent intents and
improvement opportunities discovered through trace analysis.

This implements the human-in-the-loop governance model where the system
proposes improvements but humans review and approve them.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from pathlib import Path
import json
import logging
from datetime import datetime

from .emergent_intent_analyzer import EmergentIntent

logger = logging.getLogger(__name__)


@dataclass
class PRProposal:
    """A pull request proposal"""
    pr_id: str
    title: str
    description: str
    branch_name: str

    # Changes to make
    files_to_create: List[Dict[str, str]] = field(default_factory=list)
    files_to_modify: List[Dict[str, Any]] = field(default_factory=list)
    tests_to_add: List[str] = field(default_factory=list)

    # Metadata
    priority: float = 0.5
    estimated_impact: str = "MEDIUM"
    estimated_effort: str = "MEDIUM"

    # Evidence
    supporting_data: Dict[str, Any] = field(default_factory=dict)

    # Status
    status: str = "proposed"  # proposed, drafted, ready, merged
    created_at: Optional[datetime] = None


class PRGenerator:
    """
    Generates pull request proposals for system improvements.

    Takes emergent intents and translates them into concrete PR proposals
    with file changes, test suggestions, and documentation updates.
    """

    def __init__(self, repo_path: Optional[Path] = None):
        """
        Initialize PR Generator.

        Args:
            repo_path: Path to git repository
        """
        self.repo_path = repo_path or Path.cwd()
        self.proposals: List[PRProposal] = []

        logger.info(f"PRGenerator initialized (repo={self.repo_path})")

    def generate_pr_from_intent(self, intent: EmergentIntent) -> PRProposal:
        """
        Generate a PR proposal from an emergent intent.

        Args:
            intent: Emergent intent to implement

        Returns:
            PR proposal with suggested changes
        """
        logger.info(f"Generating PR for intent: {intent.proposed_name}")

        # Create branch name
        branch_name = f"improvement/{intent.intent_id}"

        # Generate PR description
        description = self._generate_pr_description(intent)

        # Determine files to create/modify
        files_to_create = self._suggest_files_to_create(intent)
        files_to_modify = self._suggest_files_to_modify(intent)
        tests_to_add = self._suggest_tests(intent)

        # Calculate effort estimate
        effort = self._estimate_effort(len(files_to_create), len(files_to_modify))

        proposal = PRProposal(
            pr_id=f"pr_{intent.intent_id}",
            title=f"Implement {intent.proposed_name}",
            description=description,
            branch_name=branch_name,
            files_to_create=files_to_create,
            files_to_modify=files_to_modify,
            tests_to_add=tests_to_add,
            priority=self._calculate_priority(intent),
            estimated_impact=self._estimate_impact(intent),
            estimated_effort=effort,
            supporting_data={
                'intent': intent.intent_id,
                'frequency': intent.frequency,
                'confidence': intent.confidence,
                'evidence': intent.evidence
            },
            status="proposed",
            created_at=datetime.now()
        )

        self.proposals.append(proposal)

        return proposal

    def _generate_pr_description(self, intent: EmergentIntent) -> str:
        """Generate PR description markdown"""
        lines = []

        lines.append(f"## {intent.proposed_name}")
        lines.append("")
        lines.append(f"**Description:** {intent.description}")
        lines.append("")

        lines.append("### Problem")
        lines.append("")
        lines.append(f"Analysis of execution traces identified a recurring pattern:")
        lines.append(f"- **Frequency:** {intent.frequency} occurrences")
        lines.append(f"- **Confidence:** {intent.confidence:.2f}")
        lines.append("")

        if intent.evidence:
            lines.append("### Evidence")
            lines.append("")
            for key, value in intent.evidence.items():
                if isinstance(value, (str, int, float)):
                    lines.append(f"- **{key}:** {value}")
            lines.append("")

        lines.append("### Proposed Solution")
        lines.append("")
        if intent.suggested_implementation:
            lines.append(intent.suggested_implementation)
        else:
            lines.append("Implementation details to be determined during review.")
        lines.append("")

        if intent.example_queries:
            lines.append("### Example Queries")
            lines.append("")
            for query in intent.example_queries[:5]:
                lines.append(f"- {query}")
            lines.append("")

        lines.append("### Testing")
        lines.append("")
        lines.append("- [ ] Unit tests added")
        lines.append("- [ ] Integration tests passing")
        lines.append("- [ ] Manual testing complete")
        lines.append("")

        lines.append("---")
        lines.append("")
        lines.append("*This PR was automatically generated by the Learning Loop system.*")
        lines.append("*Human review and approval required before merging.*")

        return "\n".join(lines)

    def _suggest_files_to_create(self, intent: EmergentIntent) -> List[Dict[str, str]]:
        """Suggest new files to create"""
        files = []

        # If suggesting a new expert
        if "Expert" in intent.proposed_name:
            expert_name = intent.proposed_name.lower().replace("_", "")
            files.append({
                'path': f"klareco/experts/{expert_name}.py",
                'description': f"Implementation of {intent.proposed_name}",
                'template': 'expert'
            })

        # If suggesting a new tool
        elif "Tool" in intent.proposed_name or "Handler" in intent.proposed_name:
            tool_name = intent.proposed_name.lower().replace("_", "")
            files.append({
                'path': f"klareco/tools/{tool_name}.py",
                'description': f"Implementation of {intent.proposed_name}",
                'template': 'tool'
            })

        return files

    def _suggest_files_to_modify(self, intent: EmergentIntent) -> List[Dict[str, Any]]:
        """Suggest existing files to modify"""
        files = []

        # Check error type
        error_type = intent.evidence.get('error_type')

        if error_type == 'parsing_error':
            files.append({
                'path': 'klareco/parser.py',
                'reason': 'Expand vocabulary or improve parsing logic',
                'sections': ['KNOWN_ROOTS', 'parse_word']
            })

        elif error_type == 'translation_error':
            files.append({
                'path': 'klareco/front_door.py',
                'reason': 'Add fallback translation handling',
                'sections': ['FrontDoor.process']
            })

        elif error_type == 'routing_error':
            files.append({
                'path': 'klareco/orchestrator.py',
                'reason': 'Improve intent classification',
                'sections': ['Orchestrator.route']
            })

        # If improving existing expert
        if intent.suggested_expert:
            expert_file = f"klareco/experts/{intent.suggested_expert.lower()}.py"
            files.append({
                'path': expert_file,
                'reason': f'Improve {intent.suggested_expert} success rate',
                'sections': ['execute', 'can_handle']
            })

        return files

    def _suggest_tests(self, intent: EmergentIntent) -> List[str]:
        """Suggest tests to add"""
        tests = []

        # Add test file for new components
        if "Expert" in intent.proposed_name:
            expert_name = intent.proposed_name.lower().replace("_", "")
            tests.append(f"tests/test_{expert_name}.py")

        if "Tool" in intent.proposed_name:
            tool_name = intent.proposed_name.lower().replace("_", "")
            tests.append(f"tests/test_{tool_name}.py")

        # Add integration tests
        tests.append("tests/test_integration.py - Add test cases for new functionality")

        # Add specific test cases based on examples
        if intent.example_queries:
            tests.append(f"Add {len(intent.example_queries)} test cases from example queries")

        return tests

    def _calculate_priority(self, intent: EmergentIntent) -> float:
        """Calculate priority score"""
        frequency_score = min(intent.frequency / 20.0, 1.0)
        confidence_score = intent.confidence
        return (frequency_score * 0.6 + confidence_score * 0.4)

    def _estimate_impact(self, intent: EmergentIntent) -> str:
        """Estimate impact of implementation"""
        if intent.frequency >= 10:
            return "HIGH - Affects many queries"
        elif intent.frequency >= 5:
            return "MEDIUM - Affects multiple queries"
        else:
            return "LOW - Affects few queries"

    def _estimate_effort(self, files_to_create: int, files_to_modify: int) -> str:
        """Estimate implementation effort"""
        total_changes = files_to_create * 2 + files_to_modify  # Creating is harder than modifying

        if total_changes >= 6:
            return "HIGH - Multiple files and tests"
        elif total_changes >= 3:
            return "MEDIUM - Several files to change"
        else:
            return "LOW - Few changes needed"

    def generate_pr_batch(
        self,
        intents: List[EmergentIntent],
        max_prs: int = 5
    ) -> List[PRProposal]:
        """
        Generate a batch of PR proposals from multiple intents.

        Args:
            intents: List of emergent intents
            max_prs: Maximum number of PRs to generate

        Returns:
            List of PR proposals (highest priority first)
        """
        logger.info(f"Generating batch of PRs from {len(intents)} intents")

        # Generate proposals for all intents
        proposals = [self.generate_pr_from_intent(intent) for intent in intents]

        # Sort by priority
        proposals.sort(key=lambda p: p.priority, reverse=True)

        # Return top N
        return proposals[:max_prs]

    def generate_report(self) -> str:
        """Generate human-readable report of PR proposals"""
        lines = []

        lines.append("=" * 80)
        lines.append("PULL REQUEST PROPOSALS")
        lines.append("=" * 80)
        lines.append("")

        if not self.proposals:
            lines.append("No PR proposals generated.")
            lines.append("")
            lines.append("=" * 80)
            return "\n".join(lines)

        lines.append(f"Total proposals: {len(self.proposals)}")
        lines.append("")

        for i, proposal in enumerate(self.proposals, 1):
            lines.append(f"{i}. {proposal.title}")
            lines.append(f"   Branch: {proposal.branch_name}")
            lines.append(f"   Priority: {proposal.priority:.2f}")
            lines.append(f"   Impact: {proposal.estimated_impact}")
            lines.append(f"   Effort: {proposal.estimated_effort}")
            lines.append(f"   Status: {proposal.status}")
            lines.append("")

            if proposal.files_to_create:
                lines.append(f"   Files to create: {len(proposal.files_to_create)}")
                for file_info in proposal.files_to_create[:3]:
                    lines.append(f"     - {file_info['path']}")

            if proposal.files_to_modify:
                lines.append(f"   Files to modify: {len(proposal.files_to_modify)}")
                for file_info in proposal.files_to_modify[:3]:
                    lines.append(f"     - {file_info['path']}")

            lines.append("")

        lines.append("=" * 80)

        return "\n".join(lines)

    def export_proposals(self, output_file: Path):
        """Export PR proposals to JSON"""
        data = {
            'total_proposals': len(self.proposals),
            'proposals': [
                {
                    'pr_id': p.pr_id,
                    'title': p.title,
                    'description': p.description,
                    'branch_name': p.branch_name,
                    'files_to_create': p.files_to_create,
                    'files_to_modify': p.files_to_modify,
                    'tests_to_add': p.tests_to_add,
                    'priority': p.priority,
                    'estimated_impact': p.estimated_impact,
                    'estimated_effort': p.estimated_effort,
                    'supporting_data': p.supporting_data,
                    'status': p.status,
                    'created_at': p.created_at.isoformat() if p.created_at else None
                }
                for p in self.proposals
            ]
        }

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.info(f"PR proposals exported to {output_file}")

    def write_pr_markdown(self, proposal: PRProposal, output_dir: Path):
        """Write PR proposal as markdown file"""
        output_dir.mkdir(parents=True, exist_ok=True)

        filename = f"{proposal.pr_id}.md"
        filepath = output_dir / filename

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(proposal.description)

        logger.info(f"PR markdown written to {filepath}")

        return filepath


# Factory function
def create_pr_generator(repo_path: Optional[Path] = None) -> PRGenerator:
    """
    Create and return a PRGenerator instance.

    Args:
        repo_path: Path to git repository

    Returns:
        Initialized PRGenerator
    """
    return PRGenerator(repo_path=repo_path)


if __name__ == "__main__":
    # Test PR generator
    print("Testing PR Generator")
    print("=" * 80)

    from .emergent_intent_analyzer import EmergentIntent

    # Create mock emergent intents
    mock_intents = [
        EmergentIntent(
            intent_id="emergent_parsing_error_5",
            proposed_name="Extended_Vocabulary_Handler",
            description="Handle queries with unknown Esperanto words",
            frequency=5,
            confidence=0.8,
            evidence={
                'error_type': 'parsing_error',
                'total_count': 5
            },
            suggested_implementation="Expand vocabulary or implement fuzzy matching"
        ),
        EmergentIntent(
            intent_id="improve_mathexpert",
            proposed_name="Enhanced_MathExpert",
            description="Improved handling for MathExpert queries",
            frequency=8,
            confidence=0.6,
            evidence={
                'expert': 'MathExpert',
                'success_rate': 0.65,
                'count': 8
            },
            suggested_expert="MathExpert",
            suggested_implementation="Review MathExpert implementation to improve success rate"
        )
    ]

    # Generate PRs
    generator = create_pr_generator()

    proposals = generator.generate_pr_batch(mock_intents, max_prs=5)

    # Print report
    print(generator.generate_report())

    # Export proposals
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        export_file = Path(tmpdir) / "pr_proposals.json"
        generator.export_proposals(export_file)
        print(f"\n✅ Proposals exported to {export_file}")

        # Write PR markdown files
        pr_dir = Path(tmpdir) / "pr_proposals"
        for proposal in proposals:
            filepath = generator.write_pr_markdown(proposal, pr_dir)
            print(f"✅ PR markdown written to {filepath}")

    print("\n✅ PR Generator test complete!")
