#!/usr/bin/env python3
"""Find the Esperanto Wikipedia article in the database."""

import kuzu

db = kuzu.Database('data/indexes/v2.1_kuzu_index_full')
conn = kuzu.Connection(db)

# Search for Wikipedia articles about Esperanto with definitional content
result = conn.execute('''
    MATCH (d:Dokumento)-[*1..3]-(ft:Frazoteksto)
    WHERE d.metadatenoj CONTAINS 'wikipedia'
      AND ft.teksto CONTAINS 'Esperanto'
      AND ft.teksto CONTAINS 'lingvo'
      AND (ft.teksto CONTAINS 'internacia' OR ft.teksto CONTAINS 'planlingvo')
    RETURN DISTINCT d.titolo, ft.teksto
    LIMIT 20
''')

print('Wikipedia articles with definitional Esperanto content:')
print('='*70)
count = 0
while result.has_next():
    row = result.get_next()
    count += 1
    print(f'\n{count}. Article: {row[0]}')
    text = row[1][:300] if len(row[1]) > 300 else row[1]
    print(f'   Text: {text}')
