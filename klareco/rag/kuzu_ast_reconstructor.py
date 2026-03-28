"""
Kùzu AST Reconstructor

Reconstructs AST dictionaries from Kùzu graph structure.
This avoids re-parsing sentences that have already been parsed and stored.

Usage:
    reconstructor = KuzuASTReconstructor(kuzu_conn)
    ast = reconstructor.reconstruct_ast(sentence_id)
"""

import logging
from typing import Dict, List, Optional

import kuzu

logger = logging.getLogger(__name__)


class KuzuASTReconstructor:
    """Reconstruct AST dictionaries from Kùzu graph structure."""

    def __init__(self, kuzu_conn: kuzu.Connection):
        self.kuzu_conn = kuzu_conn

    def reconstruct_ast_batch(self, sentence_ids: List[int]) -> Dict[int, Dict]:
        """
        Reconstruct ASTs for multiple sentences in one query.

        Args:
            sentence_ids: List of Frazoteksto IDs

        Returns:
            Dict mapping sentence_id → AST dict
        """
        if not sentence_ids:
            return {}

        ids_str = ','.join(str(sid) for sid in sentence_ids)

        # Query to get basic AST structure
        # NOTE: This is a simplified query. Full reconstruction would need
        # recursive traversal or multiple queries for vortgrupo structure.
        query = f"""
            MATCH (ft:Frazoteksto)-[:FRAZOTEKSTO_HAVAS_AST]->(a:AST)-[:AST_HAVAS_FRAZON]->(frazo:Frazo)
            WHERE ft.id IN [{ids_str}]

            OPTIONAL MATCH (frazo)-[:HAVAS_VERBON]->(verb:Vorto)
            OPTIONAL MATCH (verb)-[:HAVAS_RADIKON]->(verb_rad:Radiko)

            OPTIONAL MATCH (frazo)-[:HAVAS_SUBJEKTON_VORTO]->(subj_v:Vorto)
            OPTIONAL MATCH (subj_v)-[:HAVAS_RADIKON]->(subj_rad:Radiko)

            OPTIONAL MATCH (frazo)-[:HAVAS_SUBJEKTON_VORTGRUPO]->(subj_vg:Vortgrupo)
            OPTIONAL MATCH (subj_vg)-[:HAVAS_KERNON]->(subj_kern:Vorto)
            OPTIONAL MATCH (subj_kern)-[:HAVAS_RADIKON]->(subj_kern_rad:Radiko)

            OPTIONAL MATCH (frazo)-[:HAVAS_OBJEKTON_VORTO]->(obj_v:Vorto)
            OPTIONAL MATCH (obj_v)-[:HAVAS_RADIKON]->(obj_rad:Radiko)

            OPTIONAL MATCH (frazo)-[:HAVAS_OBJEKTON_VORTGRUPO]->(obj_vg:Vortgrupo)
            OPTIONAL MATCH (obj_vg)-[:HAVAS_KERNON]->(obj_kern:Vorto)
            OPTIONAL MATCH (obj_kern)-[:HAVAS_RADIKON]->(obj_kern_rad:Radiko)

            RETURN ft.id, a, frazo,
                   verb, verb_rad,
                   subj_v, subj_rad,
                   subj_vg, subj_kern, subj_kern_rad,
                   obj_v, obj_rad,
                   obj_vg, obj_kern, obj_kern_rad
        """

        result = self.kuzu_conn.execute(query)

        asts = {}
        while result.has_next():
            row = result.get_next()
            sentence_id = row[0]
            ast_node = row[1]
            frazo = row[2]

            # Reconstruct AST dict
            ast = {
                'tipo': 'frazo',
                'parse_statistics': {
                    'total_words': ast_node.get('tutaj_vortoj', 0),
                    'success_rate': ast_node.get('sukcesoprocento', 0.0)
                }
            }

            # Verb (row[3] = verb Vorto, row[4] = verb Radiko)
            if row[3]:
                ast['verbo'] = self._vorto_to_dict(row[3], row[4])

            # Subject (either Vorto or Vortgrupo)
            if row[5]:  # subj_v
                ast['subjekto'] = self._vorto_to_dict(row[5], row[6])
            elif row[7]:  # subj_vg
                ast['subjekto'] = self._vortgrupo_to_dict(row[7], row[8], row[9])

            # Object (either Vorto or Vortgrupo)
            if row[10]:  # obj_v
                ast['objekto'] = self._vorto_to_dict(row[10], row[11])
            elif row[12]:  # obj_vg
                ast['objekto'] = self._vortgrupo_to_dict(row[12], row[13], row[14])

            # TODO: Reconstruct 'aliaj' (modifiers, adverbs, etc.)
            # This requires additional queries for full fidelity
            ast['aliaj'] = []

            asts[sentence_id] = ast

        return asts

    def _vorto_to_dict(self, vorto_node: Dict, radiko_node: Optional[Dict]) -> Dict:
        """Convert Vorto node to AST dict format."""
        return {
            'tipo': 'vorto',
            'plena_vorto': vorto_node.get('plena_vorto', ''),
            'radiko': vorto_node.get('radiko', ''),
            'vortspeco': vorto_node.get('vortspeco', ''),
            'kazo': vorto_node.get('kazo'),
            'nombro': vorto_node.get('nombro'),
            'tempo': vorto_node.get('tempo'),
            'modo': vorto_node.get('modo'),
            'prefiksoj': vorto_node.get('prefiksoj'),
            'sufiksoj': vorto_node.get('sufiksoj'),
        }

    def _vortgrupo_to_dict(
        self,
        vortgrupo_node: Dict,
        kerno_vorto: Optional[Dict],
        kerno_radiko: Optional[Dict]
    ) -> Dict:
        """Convert Vortgrupo node to AST dict format."""
        vg = {
            'tipo': 'vortgrupo',
            'priskriboj': [],  # TODO: Query descriptors
            'aliaj': []
        }

        if kerno_vorto:
            vg['kerno'] = self._vorto_to_dict(kerno_vorto, kerno_radiko)

        return vg

    def reconstruct_ast(self, sentence_id: int) -> Optional[Dict]:
        """Reconstruct single AST."""
        asts = self.reconstruct_ast_batch([sentence_id])
        return asts.get(sentence_id)


def has_precomputed_asts(kuzu_conn: kuzu.Connection) -> bool:
    """
    Check if Kùzu database has pre-computed ASTs.

    Returns:
        True if database has FRAZOTEKSTO_HAVAS_AST relationship
    """
    try:
        result = kuzu_conn.execute("""
            MATCH (ft:Frazoteksto)-[:FRAZOTEKSTO_HAVAS_AST]->(a:AST)
            RETURN a LIMIT 1;
        """)
        return result.has_next()
    except Exception as e:
        logger.debug(f"No pre-computed ASTs found: {e}")
        return False
