"""
Kuzu AST Reconstructor.

Rebuild parsed-AST dicts from precomputed Vorto/Vortgrupo nodes in the v2.1 graph
in a single batched query, returning primitive properties (not Kuzu node handles)
because Kuzu's node-object serialization across the Python boundary is the
dominant cost — measured 13.8s for 25 nodes vs <50ms when returning scalars.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import kuzu


_VORTO_FIELDS = (
    'plena_vorto', 'radiko', 'vortspeco', 'kazo', 'nombro',
    'tempo', 'modo', 'prefiksoj', 'sufiksoj',
)


class KuzuASTReconstructor:
    """Fetch sentence ASTs from the graph in batches without per-row overhead."""

    def __init__(self, kuzu_conn: kuzu.Connection):
        self.kuzu_conn = kuzu_conn

    def reconstruct_ast(self, sentence_id: int) -> Optional[Dict[str, Any]]:
        return self.reconstruct_ast_batch([sentence_id]).get(sentence_id)

    def reconstruct_ast_batch(self, sentence_ids: List[int]) -> Dict[int, Dict]:
        if not sentence_ids:
            return {}

        ids_str = ','.join(str(sid) for sid in sentence_ids)

        # Returning scalars (not node objects) — node serialization is what was slow.
        query = f"""
            MATCH (ft:Frazoteksto)-[:FRAZOTEKSTO_HAVAS_AST]->(a:AST)-[:AST_HAVAS_FRAZON]->(frazo:Frazo)
            WHERE ft.id IN [{ids_str}]

            OPTIONAL MATCH (frazo)-[:HAVAS_VERBON]->(verb:Vorto)
            OPTIONAL MATCH (frazo)-[:HAVAS_SUBJEKTON_VORTO]->(subj_v:Vorto)
            OPTIONAL MATCH (frazo)-[:HAVAS_SUBJEKTON_VORTGRUPO]->(subj_vg:Vortgrupo)
                            -[:HAVAS_KERNON]->(subj_kern:Vorto)
            OPTIONAL MATCH (frazo)-[:HAVAS_OBJEKTON_VORTO]->(obj_v:Vorto)
            OPTIONAL MATCH (frazo)-[:HAVAS_OBJEKTON_VORTGRUPO]->(obj_vg:Vortgrupo)
                            -[:HAVAS_KERNON]->(obj_kern:Vorto)

            RETURN
                ft.id,
                a.tutaj_vortoj, a.sukcesoprocento,
                verb.plena_vorto, verb.radiko, verb.vortspeco,
                verb.kazo, verb.nombro, verb.tempo, verb.modo,
                verb.prefiksoj, verb.sufiksoj,
                subj_v.plena_vorto, subj_v.radiko, subj_v.vortspeco,
                subj_v.kazo, subj_v.nombro, subj_v.tempo, subj_v.modo,
                subj_v.prefiksoj, subj_v.sufiksoj,
                subj_kern.plena_vorto, subj_kern.radiko, subj_kern.vortspeco,
                subj_kern.kazo, subj_kern.nombro, subj_kern.tempo, subj_kern.modo,
                subj_kern.prefiksoj, subj_kern.sufiksoj,
                obj_v.plena_vorto, obj_v.radiko, obj_v.vortspeco,
                obj_v.kazo, obj_v.nombro, obj_v.tempo, obj_v.modo,
                obj_v.prefiksoj, obj_v.sufiksoj,
                obj_kern.plena_vorto, obj_kern.radiko, obj_kern.vortspeco,
                obj_kern.kazo, obj_kern.nombro, obj_kern.tempo, obj_kern.modo,
                obj_kern.prefiksoj, obj_kern.sufiksoj
        """

        result = self.kuzu_conn.execute(query)
        asts: Dict[int, Dict[str, Any]] = {}

        while result.has_next():
            row = result.get_next()
            sentence_id = row[0]
            ast: Dict[str, Any] = {
                'tipo': 'frazo',
                'parse_statistics': {
                    'total_words':  row[1] or 0,
                    'success_rate': row[2] or 0.0,
                },
                'aliaj': [],  # not reconstructed in this fast path
            }

            verb = self._row_slice_to_vorto(row, 3)
            if verb:
                ast['verbo'] = verb

            subj_v = self._row_slice_to_vorto(row, 12)
            subj_kern = self._row_slice_to_vorto(row, 21)
            if subj_v:
                ast['subjekto'] = subj_v
            elif subj_kern:
                ast['subjekto'] = {'tipo': 'vortgrupo', 'kerno': subj_kern,
                                   'priskriboj': [], 'aliaj': []}

            obj_v = self._row_slice_to_vorto(row, 30)
            obj_kern = self._row_slice_to_vorto(row, 39)
            if obj_v:
                ast['objekto'] = obj_v
            elif obj_kern:
                ast['objekto'] = {'tipo': 'vortgrupo', 'kerno': obj_kern,
                                  'priskriboj': [], 'aliaj': []}

            asts[sentence_id] = ast

        return asts

    @staticmethod
    def _row_slice_to_vorto(row, offset: int) -> Optional[Dict[str, Any]]:
        """Materialize a 9-field Vorto slice from a row, or None if all-null."""
        plena = row[offset]
        if plena is None:
            return None
        return {
            'tipo': 'vorto',
            'plena_vorto': plena,
            'radiko':      row[offset + 1] or '',
            'vortspeco':   row[offset + 2] or '',
            'kazo':        row[offset + 3],
            'nombro':      row[offset + 4],
            'tempo':       row[offset + 5],
            'modo':        row[offset + 6],
            'prefiksoj':   row[offset + 7],
            'sufiksoj':    row[offset + 8],
        }


def has_precomputed_asts(kuzu_conn: kuzu.Connection) -> bool:
    """Check if the graph has precomputed ASTs available."""
    try:
        result = kuzu_conn.execute("""
            MATCH (a:AST) RETURN count(a) AS n LIMIT 1
        """)
        if result.has_next():
            return result.get_next()[0] > 0
    except Exception:
        return False
    return False
