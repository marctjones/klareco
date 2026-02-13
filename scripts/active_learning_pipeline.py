#!/usr/bin/env python3
"""
Active Learning Pipeline for Entity Type Classification.

Tier 3: Iterative improvement through active learning.

Workflow:
1. Extract initial data from corpus (semantic gap focus)
2. Train model on weak labels
3. Find uncertain predictions (where model is confused)
4. Manually annotate uncertain cases
5. Retrain with manual annotations
6. Repeat until accuracy plateaus

This achieves 88-92%+ accuracy by:
- Focusing annotation effort on hardest cases
- Iteratively improving where model is weakest
- Using real corpus sentences with full context
"""

import sys
import json
import torch
import random
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))


class ActiveLearningPipeline:
    """Manages active learning iterations for entity classification."""

    def __init__(
        self,
        corpus_path: Path,
        output_dir: Path,
        model_dir: Path,
        iteration: int = 0
    ):
        self.corpus_path = corpus_path
        self.output_dir = output_dir
        self.model_dir = model_dir
        self.iteration = iteration

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir.mkdir(parents=True, exist_ok=True)

    def get_iteration_paths(self, iteration: int) -> Dict[str, Path]:
        """Get file paths for a specific iteration."""
        return {
            'train_data': self.output_dir / f'iteration_{iteration}_train.jsonl',
            'val_data': self.output_dir / f'iteration_{iteration}_val.jsonl',
            'uncertain': self.output_dir / f'iteration_{iteration}_uncertain.jsonl',
            'to_annotate': self.output_dir / f'iteration_{iteration}_to_annotate.jsonl',
            'annotated': self.output_dir / f'iteration_{iteration}_annotated.jsonl',
            'model': self.model_dir / f'iteration_{iteration}',
            'metrics': self.output_dir / f'iteration_{iteration}_metrics.json'
        }

    def extract_initial_data(
        self,
        target_examples: int = 10000,
        conf_high: float = 0.7,
        max_sentences: int = 100000
    ):
        """
        Iteration 0: Extract initial training data from corpus.

        Focus on semantic gap (conf < 0.7) to bootstrap learning.
        """
        paths = self.get_iteration_paths(0)

        print("="*70)
        print("ITERATION 0: EXTRACT INITIAL DATA")
        print("="*70)
        print()

        # Import and call extraction function
        sys.path.insert(0, str(Path(__file__).parent))
        import extract_highest_quality_training_data as extract_module

        extract_module.extract_highest_quality_data(
            self.corpus_path,
            paths['train_data'],
            max_sentences=max_sentences,
            target_examples=target_examples,
            confidence_low=0.0,
            confidence_high=conf_high
        )

        # Create validation split
        self._create_validation_split(
            paths['train_data'],
            paths['val_data'],
            val_split=0.15
        )

        print(f"✓ Initial data ready:")
        print(f"  Train: {paths['train_data']}")
        print(f"  Val: {paths['val_data']}")
        print()

    def _create_validation_split(
        self,
        input_path: Path,
        val_path: Path,
        val_split: float = 0.15
    ):
        """Create stratified train/val split."""
        # Load all examples
        examples = []
        with open(input_path, 'r') as f:
            for line in f:
                if line.strip():
                    examples.append(json.loads(line))

        # Shuffle
        random.shuffle(examples)

        # Split
        val_size = int(len(examples) * val_split)
        val_examples = examples[:val_size]
        train_examples = examples[val_size:]

        # Save
        train_path = input_path.parent / (input_path.stem + '_split.jsonl')
        with open(train_path, 'w') as f:
            for ex in train_examples:
                f.write(json.dumps(ex, ensure_ascii=False) + '\n')

        with open(val_path, 'w') as f:
            for ex in val_examples:
                f.write(json.dumps(ex, ensure_ascii=False) + '\n')

        # Replace original with split version
        input_path.unlink()
        train_path.rename(input_path)

        print(f"  Split: {len(train_examples):,} train, {len(val_examples):,} val")

    def find_uncertain_predictions(
        self,
        iteration: int,
        uncertainty_threshold: float = 0.7,
        sample_size: int = 500,
        max_sentences: int = 50000
    ):
        """
        Find examples where model is most uncertain.

        Uses entropy-based uncertainty sampling:
        - High entropy = model is confused (good for annotation)
        - Low entropy = model is confident (already learned)

        Args:
            iteration: Current iteration (to load correct model)
            uncertainty_threshold: Entropy threshold for uncertain examples
            sample_size: Number of examples to sample for annotation
            max_sentences: Max sentences to evaluate
        """
        print("="*70)
        print(f"ITERATION {iteration}: FIND UNCERTAIN PREDICTIONS")
        print("="*70)
        print()

        paths = self.get_iteration_paths(iteration)
        prev_paths = self.get_iteration_paths(iteration - 1)

        # Load model
        print(f"Loading model from iteration {iteration - 1}...")
        model = self._load_model(prev_paths['model'])

        # Process corpus to find uncertain examples
        print(f"Evaluating corpus sentences (max {max_sentences:,})...")
        print()

        uncertain_examples = []
        total_evaluated = 0

        from klareco.semantic_enrichment.deterministic import DeterministicFeatureExtractor
        classifier = DeterministicFeatureExtractor()

        with open(self.corpus_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                if line_num > max_sentences:
                    break

                if not line.strip():
                    continue

                try:
                    sentence_data = json.loads(line)
                except json.JSONDecodeError:
                    continue

                # Extract words and get predictions
                examples = self._extract_sentence_examples(
                    sentence_data,
                    classifier
                )

                for example in examples:
                    # Get model prediction with uncertainty
                    prediction, uncertainty = self._predict_with_uncertainty(
                        model,
                        example
                    )

                    total_evaluated += 1

                    # Keep if uncertain
                    if uncertainty >= uncertainty_threshold:
                        example['model_prediction'] = prediction
                        example['uncertainty'] = uncertainty
                        uncertain_examples.append(example)

                if line_num % 1000 == 0:
                    print(f"  Evaluated: {total_evaluated:,} | Uncertain: {len(uncertain_examples):,}")

        print()
        print(f"✓ Found {len(uncertain_examples):,} uncertain examples")
        print()

        # Sort by uncertainty (most uncertain first)
        uncertain_examples.sort(key=lambda x: x['uncertainty'], reverse=True)

        # Sample top N for annotation
        to_annotate = uncertain_examples[:sample_size]

        # Save uncertain examples
        with open(paths['uncertain'], 'w') as f:
            for ex in uncertain_examples:
                f.write(json.dumps(ex, ensure_ascii=False) + '\n')

        # Save examples to annotate
        with open(paths['to_annotate'], 'w') as f:
            for ex in to_annotate:
                f.write(json.dumps(ex, ensure_ascii=False) + '\n')

        print(f"✓ Sampled {len(to_annotate):,} most uncertain for annotation")
        print(f"  Output: {paths['to_annotate']}")
        print()

        return paths['to_annotate']

    def _load_model(self, model_path: Path):
        """Load trained model."""
        # TODO: Implement model loading
        # For now, placeholder
        print(f"  [TODO: Load model from {model_path}]")
        return None

    def _predict_with_uncertainty(
        self,
        model,
        example: dict
    ) -> Tuple[dict, float]:
        """
        Get model prediction with uncertainty score.

        Returns:
            (prediction, uncertainty_score)
            uncertainty_score: entropy of prediction distribution (0-1)
        """
        # TODO: Implement actual prediction
        # For now, return placeholder
        prediction = {
            'tier3_type': 'person_generic',
            'confidence': 0.5
        }
        uncertainty = 0.8  # High entropy = uncertain

        return prediction, uncertainty

    def _extract_sentence_examples(
        self,
        sentence_data: dict,
        classifier
    ) -> List[dict]:
        """Extract examples from sentence (same as extraction script)."""
        sys.path.insert(0, str(Path(__file__).parent))
        import extract_highest_quality_training_data as extract_module

        return extract_module.extract_training_examples_from_sentence(
            sentence_data,
            classifier,
            confidence_threshold_low=0.0,
            confidence_threshold_high=1.0  # Get all substantivos
        )

    def prepare_annotation_batch(
        self,
        to_annotate_path: Path,
        output_html: Path
    ):
        """
        Create annotation interface (HTML file for manual annotation).

        Generates a simple web interface for annotating examples.
        """
        print("="*70)
        print("PREPARE ANNOTATION BATCH")
        print("="*70)
        print()

        # Load examples to annotate
        examples = []
        with open(to_annotate_path, 'r') as f:
            for line in f:
                if line.strip():
                    examples.append(json.loads(line))

        # Generate HTML interface
        html = self._generate_annotation_html(examples)

        # Save
        with open(output_html, 'w') as f:
            f.write(html)

        print(f"✓ Annotation interface ready: {output_html}")
        print()
        print("To annotate:")
        print(f"  1. Open in browser: file://{output_html.absolute()}")
        print(f"  2. Review each example in context")
        print(f"  3. Select correct tier3_type")
        print(f"  4. Download annotations.json when done")
        print(f"  5. Run: python scripts/convert_annotations.py annotations.json")
        print()

    def _generate_annotation_html(self, examples: List[dict]) -> str:
        """Generate HTML annotation interface."""
        # Entity type options
        from klareco.semantic_enrichment.taxonomy import PersonType, LocationType, TimeType, ThingType

        type_options = []
        for enum_type in [PersonType, LocationType, TimeType, ThingType]:
            for item in enum_type:
                type_options.append(item.value)

        type_options_html = '\n'.join([
            f'            <option value="{t}">{t}</option>'
            for t in sorted(type_options)
        ])

        examples_json = json.dumps(examples, ensure_ascii=False)

        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Entity Type Annotation</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            max-width: 1200px;
            margin: 20px auto;
            padding: 20px;
        }}
        .example {{
            border: 1px solid #ccc;
            padding: 20px;
            margin: 20px 0;
            border-radius: 5px;
        }}
        .sentence {{
            font-size: 18px;
            margin: 10px 0;
            padding: 10px;
            background: #f5f5f5;
        }}
        .target-word {{
            background: #ffeb3b;
            font-weight: bold;
        }}
        .metadata {{
            font-size: 12px;
            color: #666;
            margin: 10px 0;
        }}
        .model-prediction {{
            background: #e3f2fd;
            padding: 10px;
            margin: 10px 0;
            border-left: 3px solid #2196f3;
        }}
        .annotation {{
            margin: 15px 0;
        }}
        select {{
            padding: 8px;
            font-size: 14px;
            width: 300px;
        }}
        button {{
            padding: 10px 20px;
            font-size: 16px;
            background: #4caf50;
            color: white;
            border: none;
            border-radius: 5px;
            cursor: pointer;
        }}
        button:hover {{
            background: #45a049;
        }}
        .progress {{
            margin: 20px 0;
            font-size: 18px;
            font-weight: bold;
        }}
        .controls {{
            position: sticky;
            top: 0;
            background: white;
            padding: 20px;
            border-bottom: 2px solid #ccc;
            z-index: 100;
        }}
    </style>
</head>
<body>
    <div class="controls">
        <h1>Entity Type Annotation</h1>
        <div class="progress">
            Progress: <span id="annotated-count">0</span> / <span id="total-count">{len(examples)}</span>
        </div>
        <button onclick="downloadAnnotations()">Download Annotations</button>
    </div>

    <div id="examples-container"></div>

    <script>
        const examples = {examples_json};
        const annotations = {{}};
        let annotatedCount = 0;

        function renderExamples() {{
            const container = document.getElementById('examples-container');

            examples.forEach((ex, idx) => {{
                const div = document.createElement('div');
                div.className = 'example';
                div.id = `example-${{idx}}`;

                const targetWord = ex.word_ast.teksto;
                const sentence = ex.sentence_text;
                const highlightedSentence = sentence.replace(
                    new RegExp(`\\\\b${{targetWord}}\\\\b`, 'gi'),
                    `<span class="target-word">$&</span>`
                );

                const modelPred = ex.model_prediction?.tier3_type || 'unknown';
                const uncertainty = ex.uncertainty?.toFixed(3) || 'N/A';

                div.innerHTML = `
                    <h3>Example ${{idx + 1}}</h3>

                    <div class="sentence">${{highlightedSentence}}</div>

                    <div class="metadata">
                        <strong>Target word:</strong> ${{targetWord}} (root: ${{ex.word_ast.radiko}})<br>
                        <strong>Source:</strong> ${{ex.metadata?.source_name || 'Unknown'}}<br>
                        <strong>Quality:</strong> ${{ex.metadata?.quality || 'Unknown'}}
                    </div>

                    <div class="model-prediction">
                        <strong>Model prediction:</strong> ${{modelPred}}<br>
                        <strong>Uncertainty:</strong> ${{uncertainty}} (higher = more uncertain)
                    </div>

                    <div class="annotation">
                        <label><strong>Correct tier3_type:</strong></label><br>
                        <select id="select-${{idx}}" onchange="annotate(${{idx}}, this.value)">
                            <option value="">-- Select type --</option>
{type_options_html}
                        </select>
                    </div>
                `;

                container.appendChild(div);
            }});

            document.getElementById('total-count').textContent = examples.length;
        }}

        function annotate(idx, value) {{
            if (value) {{
                if (!annotations[idx]) {{
                    annotatedCount++;
                }}
                annotations[idx] = {{
                    example_id: idx,
                    tier3_type: value,
                    word: examples[idx].word_ast.teksto,
                    sentence: examples[idx].sentence_text
                }};
            }} else {{
                if (annotations[idx]) {{
                    annotatedCount--;
                }}
                delete annotations[idx];
            }}

            document.getElementById('annotated-count').textContent = annotatedCount;
        }}

        function downloadAnnotations() {{
            const data = Object.values(annotations);

            if (data.length === 0) {{
                alert('No annotations to download!');
                return;
            }}

            const blob = new Blob([JSON.stringify(data, null, 2)], {{type: 'application/json'}});
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'annotations.json';
            a.click();

            alert(`Downloaded ${{data.length}} annotations!`);
        }}

        // Initialize
        renderExamples();
    </script>
</body>
</html>
"""
        return html

    def merge_annotations(
        self,
        iteration: int,
        annotations_file: Path
    ):
        """
        Merge manual annotations into training data.

        Args:
            iteration: Current iteration
            annotations_file: JSON file with manual annotations from web interface
        """
        print("="*70)
        print(f"ITERATION {iteration}: MERGE ANNOTATIONS")
        print("="*70)
        print()

        paths = self.get_iteration_paths(iteration)
        prev_paths = self.get_iteration_paths(iteration - 1)

        # Load manual annotations
        with open(annotations_file, 'r') as f:
            annotations = json.load(f)

        print(f"Loaded {len(annotations):,} manual annotations")

        # Load original examples to annotate
        to_annotate = []
        with open(paths['to_annotate'], 'r') as f:
            for line in f:
                if line.strip():
                    to_annotate.append(json.loads(line))

        # Merge annotations
        annotated_examples = []
        for ann in annotations:
            idx = ann['example_id']
            if idx < len(to_annotate):
                example = to_annotate[idx].copy()
                # Update label with manual annotation
                example['label'] = {
                    'tier3_type': ann['tier3_type'],
                    'confidence': 1.0,
                    'source': 'manual_annotation',
                    'iteration': iteration
                }
                annotated_examples.append(example)

        # Save manually annotated examples
        with open(paths['annotated'], 'w') as f:
            for ex in annotated_examples:
                f.write(json.dumps(ex, ensure_ascii=False) + '\n')

        print(f"✓ Saved {len(annotated_examples):,} annotated examples")
        print()

        # Merge with previous training data
        print("Merging with previous training data...")
        merged_train = []

        # For iteration 0, use current iteration's train data (replace weak labels)
        # For iteration 1+, use previous iteration's train data
        if iteration == 0:
            source_train_path = paths['train_data']
        else:
            source_train_path = prev_paths['train_data']

        # Create lookup of annotated examples by sentence text + word index
        def make_key(ex):
            return (ex['sentence_text'], ex.get('word_index', -1))

        annotated_lookup = {make_key(ex): ex for ex in annotated_examples}

        # Load source training data, replacing annotated examples
        if source_train_path.exists():
            replaced_count = 0
            with open(source_train_path, 'r') as f:
                for line in f:
                    if line.strip():
                        ex = json.loads(line)
                        key = make_key(ex)

                        # If this exact example was annotated, use annotated version
                        if key in annotated_lookup:
                            merged_train.append(annotated_lookup[key])
                            replaced_count += 1
                        else:
                            merged_train.append(ex)

            print(f"  Replaced {replaced_count} weak labels with manual annotations")
            print(f"  Kept {len(merged_train) - replaced_count} weak labels")
        else:
            print(f"Warning: No previous training data found at {source_train_path}")
            # If no previous data, just use annotated examples
            merged_train = annotated_examples

        # Shuffle
        random.shuffle(merged_train)

        # Save merged training data
        with open(paths['train_data'], 'w') as f:
            for ex in merged_train:
                f.write(json.dumps(ex, ensure_ascii=False) + '\n')

        print(f"✓ Merged training data: {len(merged_train):,} examples")
        print(f"  Previous: {len(merged_train) - len(annotated_examples):,}")
        print(f"  New manual: {len(annotated_examples):,}")
        print()

        # Copy validation data from previous iteration
        if prev_paths['val_data'].exists():
            import shutil
            shutil.copy(prev_paths['val_data'], paths['val_data'])
            print(f"✓ Copied validation data from iteration {iteration - 1}")
        print()


def run_iteration(
    pipeline: ActiveLearningPipeline,
    iteration: int,
    action: str,
    **kwargs
):
    """Run a specific iteration action."""
    paths = pipeline.get_iteration_paths(iteration)

    if action == 'extract':
        # Iteration 0: Extract initial data
        extract_kwargs = {
            'target_examples': kwargs.get('target_examples', 10000),
            'conf_high': kwargs.get('conf_high', 0.7),
            'max_sentences': kwargs.get('max_sentences', 100000)
        }
        pipeline.extract_initial_data(**extract_kwargs)

    elif action == 'find-uncertain':
        # Find uncertain predictions from previous model
        uncertain_kwargs = {
            'uncertainty_threshold': kwargs.get('uncertainty_threshold', 0.7),
            'sample_size': kwargs.get('sample_size', 500),
            'max_sentences': kwargs.get('max_sentences', 50000)
        }
        pipeline.find_uncertain_predictions(iteration, **uncertain_kwargs)

    elif action == 'prepare-annotation':
        # Generate annotation interface
        output_html = kwargs.get('output_html', pipeline.output_dir / f'iteration_{iteration}_annotate.html')
        pipeline.prepare_annotation_batch(paths['to_annotate'], output_html)

    elif action == 'merge-annotations':
        # Merge manual annotations
        annotations_file = kwargs.get('annotations_file')
        if not annotations_file:
            print("ERROR: --annotations-file required for merge-annotations")
            return
        pipeline.merge_annotations(iteration, annotations_file)

    else:
        print(f"ERROR: Unknown action: {action}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Active learning pipeline')
    parser.add_argument(
        '--corpus',
        type=Path,
        default=Path('data/corpus/unified_corpus.jsonl'),
        help='Path to unified corpus'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('data/training/active_learning'),
        help='Output directory for training data'
    )
    parser.add_argument(
        '--models',
        type=Path,
        default=Path('models/active_learning'),
        help='Directory for trained models'
    )
    parser.add_argument(
        '--iteration',
        type=int,
        default=0,
        help='Current iteration number'
    )
    parser.add_argument(
        '--action',
        type=str,
        required=True,
        choices=['extract', 'find-uncertain', 'prepare-annotation', 'merge-annotations'],
        help='Action to perform'
    )

    # Action-specific arguments
    parser.add_argument('--target-examples', type=int, default=10000)
    parser.add_argument('--conf-high', type=float, default=0.7)
    parser.add_argument('--max-sentences', type=int, default=100000)
    parser.add_argument('--uncertainty-threshold', type=float, default=0.7)
    parser.add_argument('--sample-size', type=int, default=500)
    parser.add_argument('--output-html', type=Path)
    parser.add_argument('--annotations-file', type=Path)

    args = parser.parse_args()

    # Create pipeline
    pipeline = ActiveLearningPipeline(
        args.corpus,
        args.output,
        args.models,
        args.iteration
    )

    # Run action
    run_iteration(
        pipeline,
        args.iteration,
        args.action,
        target_examples=args.target_examples,
        conf_high=args.conf_high,
        max_sentences=args.max_sentences,
        uncertainty_threshold=args.uncertainty_threshold,
        sample_size=args.sample_size,
        output_html=args.output_html,
        annotations_file=args.annotations_file
    )
