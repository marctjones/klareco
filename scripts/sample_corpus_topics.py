#!/usr/bin/env python3
"""
Sample diverse topics from corpus to create test questions

Searches for factual sentences about various topics (science, history, sports, etc.)
to identify good candidates for QA test questions.
"""

import kuzu

# Topics to search
TOPICS = {
    'American Presidents': [
        ('Washington', ['prezident', 'usona', 'amerika']),
        ('Lincoln', ['prezident', 'milito', 'sklaveco']),
        ('Jefferson', ['prezident', 'amerika']),
        ('Roosevelt', ['prezident', 'usona']),
        ('Kennedy', ['prezident', 'usona'])
    ],
    'Scientists': [
        ('Einstein', ['fiziko', 'relativeco', 'scienculo']),
        ('Newton', ['fiziko', 'gravito', 'matematiko']),
        ('Curie', ['radio', 'kemio', 'premio nobel']),
        ('Darwin', ['evolucio', 'specio', 'biologio']),
        ('Galileo', ['astronomo', 'teleskopo'])
    ],
    'Inventors/Innovators': [
        ('Franklin', ['elektro', 'amerika', 'inventis']),
        ('Edison', ['lampo', 'inventis', 'elektro']),
        ('Tesla', ['elektro', 'inventis']),
        ('Bell', ['telefono', 'inventis'])
    ],
    'Writers': [
        ('Shakespeare', ['verkisto', 'angla', 'teatro']),
        ('Dickens', ['verkisto', 'angla', 'romano']),
        ('Tolstoy', ['verkisto', 'rusa']),
        ('Cervantes', ['verkisto', 'hispana'])
    ],
    'Sports': [
        ('olimpiko', ['sportoj', 'ludoj', 'konkurso']),
        ('futbalo', ['sporto', 'ludo', 'teamo']),
        ('basketbalo', ['sporto', 'ludo'])
    ]
}

def main():
    db = kuzu.Database('data/indexes/v2.1_kuzu_index_full')
    conn = kuzu.Connection(db)

    all_facts = []

    for category, topics in TOPICS.items():
        print(f'\n{"="*80}')
        print(f'{category}')
        print(f'{"="*80}')

        for name, context_words in topics:
            print(f'\n{name}:')

            # Search for sentences containing the name
            result = conn.execute(f'''
            MATCH (f:Frazoteksto)
            WHERE f.teksto CONTAINS '{name}'
            RETURN f.teksto
            LIMIT 20
            ''')

            sentences = []
            while result.has_next():
                sentence = result.get_next()[0]
                sentences.append(sentence)

            if not sentences:
                print(f'  No sentences found')
                continue

            # Filter to sentences with context words (factual info)
            factual = []
            for sent in sentences:
                sent_lower = sent.lower()
                # Check if contains any context word
                if any(word in sent_lower for word in context_words):
                    factual.append(sent)

            if factual:
                print(f'  Found {len(factual)} factual sentences:')
                for i, sent in enumerate(factual[:3], 1):
                    print(f'    [{i}] {sent[:150]}...')
                    all_facts.append({
                        'category': category,
                        'topic': name,
                        'sentence': sent
                    })
            else:
                print(f'  Found {len(sentences)} mentions, but no factual context')
                # Show first mention anyway
                print(f'    [1] {sentences[0][:150]}...')
                all_facts.append({
                    'category': category,
                    'topic': name,
                    'sentence': sentences[0]
                })

    # Summary
    print(f'\n\n{"="*80}')
    print(f'SUMMARY')
    print(f'{"="*80}')
    print(f'Total factual sentences found: {len(all_facts)}')
    print(f'\nBy category:')
    by_cat = {}
    for fact in all_facts:
        cat = fact['category']
        by_cat[cat] = by_cat.get(cat, 0) + 1

    for cat, count in by_cat.items():
        print(f'  {cat}: {count}')

if __name__ == '__main__':
    main()
