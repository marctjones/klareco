#!/usr/bin/env python3
"""
Quick extraction test - just test answer extraction on a few questions.
No full RAG pipeline, just parser + extractor.
"""

import json
import logging
from pathlib import Path
from klareco.parser import parse
from klareco.rag.answer_extractor import ASTAnswerExtractor

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def test_extraction(extractor, query, doc, expected_keywords):
    """Test extraction on a single query-document pair."""
    query_ast = parse(query)
    doc_ast = parse(doc)
    
    result = extractor.extract_answer(query_ast, doc_ast, doc)
    
    if result:
        answer = result['text'].lower()
        # Check if answer contains any expected keyword
        is_correct = any(kw.lower() in answer for kw in expected_keywords)
        return {
            'extracted': result['text'],
            'method': result['method'],
            'confidence': result['confidence'],
            'correct': is_correct,
            'expected': expected_keywords,
        }
    else:
        return {
            'extracted': None,
            'correct': False,
            'expected': expected_keywords,
        }


def main():
    logger.info("="*60)
    logger.info("Quick Extraction Test")
    logger.info("="*60)
    logger.info("")
    
    extractor = ASTAnswerExtractor()
    
    # Test cases (query, doc, expected_keywords)
    test_cases = [
        {
            'id': 1,
            'query': 'Kio estas Esperanto?',
            'doc': 'Esperanto estas unua internacia planlingvo.',
            'expected': ['planlingvo', 'lingvo'],
            'name': 'WHAT with ordinal (should skip "unua")',
        },
        {
            'id': 2,
            'query': 'Kiu fondis Esperanton?',
            'doc': 'Zamenhof fondis Esperanton en 1887.',
            'expected': ['zamenhof'],
            'name': 'WHO simple',
        },
        {
            'id': 3,
            'query': 'Kiam estis fondita UEA?',
            'doc': 'Zamenhof kreis Esperanton en 1887, kaj poste en 1908 estis fondita Universala Esperanto-Asocio.',
            'expected': ['1908'],
            'name': 'WHEN complex (multi-event, subclause scoring)',
        },
        {
            'id': 4,
            'query': 'Kie estas Varsovio?',
            'doc': 'Varsovio estas en Pollando.',
            'expected': ['pollando', 'polland', 'poland'],
            'name': 'WHERE simple',
        },
        {
            'id': 5,
            'query': 'Kiom da personoj partoprenis?',
            'doc': 'Dek personoj partoprenis en la kongreso.',
            'expected': ['dek', '10'],
            'name': 'HOW_MANY',
        },
        {
            'id': 6,
            'query': 'Kio estas planlingvo?',
            'doc': 'Planlingvo estas lingvo kreita de homo.',
            'expected': ['lingvo'],
            'name': 'WHAT definition',
        },
        {
            'id': 7,
            'query': 'Kiu parolis?',
            'doc': 'Mi parolis hieraŭ.',
            'expected': [],  # Should reject pronoun "Mi"
            'name': 'WHO with pronoun (should reject)',
        },
    ]
    
    results = []
    correct = 0
    total = 0
    
    for test in test_cases:
        result = test_extraction(
            extractor,
            test['query'],
            test['doc'],
            test['expected']
        )
        
        total += 1
        if result['correct'] or (not result['extracted'] and not test['expected']):
            correct += 1
            status = "✓"
        else:
            status = "✗"
        
        logger.info(f"[{test['id']}] {status} {test['name']}")
        logger.info(f"    Query: {test['query']}")
        logger.info(f"    Doc: {test['doc'][:60]}...")
        if result['extracted']:
            logger.info(f"    Extracted: '{result['extracted']}' ({result['method']}, conf={result['confidence']:.2f})")
        else:
            logger.info(f"    Extracted: None")
        logger.info(f"    Expected: {test['expected']}")
        logger.info("")
        
        results.append({
            'test_id': test['id'],
            'test_name': test['name'],
            'query': test['query'],
            'extracted': result['extracted'],
            'expected': test['expected'],
            'correct': result['correct'],
        })
    
    # Summary
    logger.info("="*60)
    logger.info(f"Results: {correct}/{total} correct ({100*correct/total:.1f}%)")
    logger.info("="*60)
    
    # Save results
    output_path = Path('data/evaluation/extraction_quick_test.json')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump({
            'total': total,
            'correct': correct,
            'accuracy': correct / total,
            'results': results,
        }, f, indent=2)
    
    logger.info(f"\nResults saved to: {output_path}")


if __name__ == '__main__':
    main()
