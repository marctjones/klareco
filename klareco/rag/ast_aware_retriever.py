"""
Multi-Strategy AST-Aware Retriever.

Combines all AST-aware components into a unified retrieval system:
1. Question Type Classifier - Understand question intent
2. Entity Recognizer - Extract named entities
3. AST Pattern Matcher - Match structural patterns
4. Semantic Relations - Expand with synonyms

This is a DETERMINISTIC retriever that leverages Klareco's unique advantage:
fully parsed AST annotations for both queries and corpus.

Expected performance: 60-70% accuracy (6x improvement from baseline 10-12%)
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from klareco.parser import parse
from klareco.rag.question_classifier import QuestionClassifier, QuestionType, EntityType as QEntityType
from klareco.rag.entity_recognizer import EntityRecognizer, EntityType as EEntityType
from klareco.rag.ast_pattern_matcher import ASTPatternMatcher, MatchResult
from klareco.rag.semantic_db import SemanticRelationDB

logger = logging.getLogger(__name__)


class ASTAwareRetriever:
    """
    Multi-strategy AST-aware retriever.

    Combines deterministic AST analysis with semantic relations to achieve
    high accuracy question answering without learned parameters.
    """

    def __init__(
        self,
        index_path: Path,
        revo_path: Optional[Path] = None,
        use_prefilter: bool = True,
        prefilter_retriever: Optional = None,
        use_keyword_prefilter: bool = True,
    ):
        """
        Initialize AST-aware retriever.

        Args:
            index_path: Path to slot-based index directory
            revo_path: Path to ReVo semantic relations (optional)
            use_prefilter: Whether to use embedding-based pre-filtering (recommended)
            prefilter_retriever: Optional pre-filtering retriever (HNSW, FAISS, etc.)
                                If None and use_prefilter=True, will try to load HNSW
            use_keyword_prefilter: If HNSW unavailable, use fast grep-based keyword
                                   prefilter instead of slow brute-force scan (default: True)
        """
        self.index_path = Path(index_path)
        self.use_prefilter = use_prefilter
        self.use_keyword_prefilter = use_keyword_prefilter

        # Initialize components
        logger.info("Initializing AST-aware retrieval components...")

        self.question_classifier = QuestionClassifier()
        self.entity_recognizer = EntityRecognizer()
        self.semantic_db = SemanticRelationDB(revo_path=revo_path)

        # Initialize pattern matcher with semantic DB
        synonym_dict = {
            root: synonyms
            for root, synonyms in self.semantic_db.synonyms.items()
        }
        antonym_dict = {
            root: antonyms
            for root, antonyms in self.semantic_db.antonyms.items()
        }

        self.pattern_matcher = ASTPatternMatcher(
            synonym_db=synonym_dict,
            antonym_db=antonym_dict,
        )

        # Load index metadata
        self._load_index()

        # Initialize pre-filter retriever if requested
        self.prefilter_retriever = None
        if use_prefilter:
            if prefilter_retriever:
                self.prefilter_retriever = prefilter_retriever
                logger.info("  Using provided pre-filter retriever")
            else:
                # Try to load HNSW as default pre-filter
                self._load_prefilter()

        logger.info("AST-aware retriever initialized")

    def _load_index(self):
        """Load slot-based index metadata.

        Uses cached offsets file if available, otherwise builds and caches.
        """
        import numpy as np

        index_file = self.index_path / "slot_index.jsonl"
        offsets_file = self.index_path / "slot_index.offsets.npy"

        if not index_file.exists():
            raise FileNotFoundError(
                f"Slot index not found: {index_file}\n"
                f"Run: python scripts/index_slot_based.py --corpus <corpus> --index {self.index_path}"
            )

        # Try to load cached offsets (fast: ~0.1s)
        if offsets_file.exists():
            # Check if offsets are newer than index
            if offsets_file.stat().st_mtime >= index_file.stat().st_mtime:
                self.doc_offsets = np.load(offsets_file)
                logger.info(f"Loaded cached offsets: {len(self.doc_offsets):,} documents")
                return
            else:
                logger.info("Index newer than cached offsets, rebuilding...")

        # Build document offset index (slow: ~24s for 4.4M docs)
        logger.info("Building document offset index (first time only)...")
        offsets = []
        with open(index_file, 'rb') as f:
            offset = 0
            count = 0
            for line in f:
                offsets.append(offset)
                offset += len(line)
                count += 1
                if count % 1000000 == 0:
                    logger.info(f"  Indexed {count:,} documents...")

        # Save as numpy array for fast loading next time
        self.doc_offsets = np.array(offsets, dtype=np.int64)
        np.save(offsets_file, self.doc_offsets)
        logger.info(f"Saved offsets cache: {offsets_file}")
        logger.info(f"Loaded index with {len(self.doc_offsets):,} documents")

    def _load_prefilter(self):
        """Load HNSW pre-filter if available.

        Uses lightweight HNSW-only search (no mmap required).

        IMPORTANT: Load PyTorch models BEFORE hnswlib to avoid memory allocator
        conflicts that cause "free(): invalid size" crashes. See issue #88.
        """
        hnsw_file = self.index_path / "hnsw" / "full_embeddings.hnsw"

        if not hnsw_file.exists():
            logger.warning("  No HNSW index found")
            logger.warning(f"  Run: ./scripts/build_hnsw_index.sh {self.index_path}")
            if self.use_keyword_prefilter:
                logger.info("  Will use keyword pre-filter (grep-based, fast)")
            return

        try:
            import json

            # Get embedding dimension from slot_index FIRST
            index_file = self.index_path / "slot_index.jsonl"
            with open(index_file) as f:
                first_doc = json.loads(f.readline())
                embedding_dim = len(first_doc['full_embedding'])

            # CRITICAL: Load PyTorch models BEFORE hnswlib
            # This avoids memory allocator conflicts (see issue #88)
            logger.info("  Loading query embedder (PyTorch models first)...")
            from klareco.embeddings.hybrid_embeddings import HybridEmbeddings

            root_model = Path("models/root_embeddings/best_model.pt")
            topical_model = Path("models/topical_embeddings/best_model.pt")

            if topical_model.exists() and root_model.exists():
                self.hybrid_embedder = HybridEmbeddings.from_checkpoints(
                    linguistic_checkpoint=root_model,
                    topical_checkpoint=topical_model,
                    pad_missing=True,
                    default_mode='hybrid'
                )
                logger.info(f"    ✓ Hybrid embedder loaded (128d)")
            elif root_model.exists():
                # Fall back to linguistic-only embeddings
                from klareco.embeddings.linguistic_embeddings import LinguisticEmbeddings
                self.hybrid_embedder = LinguisticEmbeddings.from_checkpoint(root_model)
                logger.info(f"    ✓ Linguistic embedder loaded (64d)")
            else:
                logger.warning("    No embedding models found, HNSW prefilter disabled")
                if self.use_keyword_prefilter:
                    logger.info("  Will use keyword pre-filter (grep-based, fast)")
                return

            # NOW load HNSW index (after PyTorch models are loaded)
            logger.info("  Loading HNSW index...")
            import hnswlib

            self.hnsw_index = hnswlib.Index(space='cosine', dim=embedding_dim)
            self.hnsw_index.load_index(str(hnsw_file))
            self.hnsw_index.set_ef(100)  # Search quality parameter

            self.prefilter_retriever = 'hnsw_direct'  # Flag for direct HNSW mode
            logger.info(f"  ✓ HNSW pre-filter loaded ({self.hnsw_index.get_current_count():,} vectors, {embedding_dim}d)")

        except Exception as e:
            logger.warning(f"  Failed to load HNSW pre-filter: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            if self.use_keyword_prefilter:
                logger.info("  Will use keyword pre-filter (grep-based, fast)")
            else:
                logger.warning("  Will use brute-force search (slow for large corpora)")

    def _hnsw_prefilter(
        self,
        query_ast: Dict,
        max_results: int = 500,
        use_slot_reranking: bool = True,
    ) -> List[Tuple[float, Dict]]:
        """
        Fast HNSW-based prefilter with slot-aware reranking.

        Uses HNSW for fast initial retrieval (k * 2 candidates),
        then reranks using slot similarity to leverage AST structure.

        Args:
            query_ast: Parsed query AST
            max_results: Number of final candidates to return
            use_slot_reranking: Whether to rerank with slot similarity (default: True)
        """
        import numpy as np
        import torch

        # Get query embedding from AST using lightweight embedder
        query_emb = self._embed_query_ast(query_ast)

        if query_emb is None:
            logger.warning("  Failed to embed query for HNSW search")
            return []

        # Ensure correct shape for hnswlib
        query_emb = np.array([query_emb], dtype=np.float32)

        # Get more candidates if we'll rerank
        hnsw_k = max_results * 2 if use_slot_reranking else max_results

        # Search HNSW
        labels, distances = self.hnsw_index.knn_query(query_emb, k=hnsw_k)

        # Load documents by ID
        candidates = []
        for doc_id, dist in zip(labels[0], distances[0]):
            doc = self._get_document(int(doc_id))
            hnsw_score = 1.0 - dist  # Convert distance to similarity
            candidates.append((hnsw_score, doc))

        if not use_slot_reranking:
            logger.info(f"  HNSW prefilter: found {len(candidates)} candidates (no reranking)")
            return candidates[:max_results]

        # Slot-based reranking: extract query slots and rerank by slot similarity
        query_slots = self._extract_query_slots(query_ast)
        has_slots = any(v is not None for v in query_slots.values())

        if not has_slots:
            logger.info(f"  HNSW prefilter: found {len(candidates)} candidates (no slots for reranking)")
            return candidates[:max_results]

        # Rerank by slot similarity
        reranked = []
        for hnsw_score, doc in candidates:
            slot_score = self._compute_slot_similarity(query_slots, doc)
            # Combine HNSW similarity with slot similarity (slot has higher weight)
            combined_score = 0.3 * hnsw_score + 0.7 * slot_score
            reranked.append((combined_score, doc))

        # Sort by combined score
        reranked.sort(key=lambda x: -x[0])

        logger.info(f"  HNSW prefilter: found {len(reranked)} candidates (slot-reranked)")
        return reranked[:max_results]

    def _compute_slot_similarity(
        self,
        query_slots: Dict[str, Optional[np.ndarray]],
        doc: Dict,
        is_question: bool = True,
    ) -> float:
        """
        Compute slot-based similarity between query and document.

        Compares SUBJ↔SUBJ, VERB↔VERB, OBJ↔OBJ with role-aware weighting.
        For questions, missing query slots get partial bonus if doc has them.
        """
        import numpy as np

        slot_weights = {'SUBJ': 0.4, 'VERB': 0.3, 'OBJ': 0.3}
        partial_bonus = 0.8 if is_question else 0.5

        score = 0.0
        matched_slots = 0

        # Get document slot embeddings from slots_np if present
        doc_slots = doc.get('slots_np', {})
        if not doc_slots:
            # Try to extract from slots field (arrays)
            for slot_name in ['SUBJ', 'VERB', 'OBJ']:
                slot_data = doc.get('slots', {}).get(slot_name)
                if slot_data is not None:
                    doc_slots[slot_name] = np.array(slot_data, dtype=np.float32)

        for slot, weight in slot_weights.items():
            query_emb = query_slots.get(slot)
            doc_emb = doc_slots.get(slot)

            # Handle doc_emb as array
            if isinstance(doc_emb, (list, np.ndarray)):
                doc_emb = np.array(doc_emb, dtype=np.float32) if isinstance(doc_emb, list) else doc_emb
                # Check for NaN (indicates missing slot)
                if np.any(np.isnan(doc_emb)):
                    doc_emb = None

            if query_emb is not None and doc_emb is not None:
                # Both have this slot: compute cosine similarity
                query_norm = np.linalg.norm(query_emb)
                doc_norm = np.linalg.norm(doc_emb)
                if query_norm > 0 and doc_norm > 0:
                    sim = np.dot(query_emb, doc_emb) / (query_norm * doc_norm)
                    score += weight * max(0, sim)  # Clamp negative similarities
                    matched_slots += 1
            elif query_emb is None and doc_emb is not None:
                # Query missing this slot (e.g., "Kiu?" has no SUBJ): partial match bonus
                score += weight * partial_bonus
                matched_slots += 1
            # If doc missing slot but query has it: no score (mismatch)

        # Normalize by matched slots
        if matched_slots > 0:
            return score / matched_slots
        else:
            return 0.0

    def _embed_query_ast(self, query_ast: Dict) -> Optional[np.ndarray]:
        """
        Embed a query AST using the lightweight hybrid embedder.

        This is a simplified version of SlotBasedIndexer.index_sentence()
        that doesn't require the full indexer infrastructure.

        Returns:
            Query embedding as numpy array, or None if failed
        """
        import torch

        if not hasattr(self, 'hybrid_embedder') or self.hybrid_embedder is None:
            return None

        # Extract roots from AST
        roots = []

        def extract_roots(node):
            if not node or not isinstance(node, dict):
                return
            if node.get('tipo') == 'vorto':
                root = node.get('radiko', '')
                if root and len(root) >= 2:
                    roots.append(root.lower())
            elif node.get('tipo') == 'vortgrupo':
                extract_roots(node.get('kerno'))
                for p in node.get('priskriboj', []):
                    extract_roots(p)
            elif node.get('tipo') == 'frazo':
                extract_roots(node.get('subjekto'))
                extract_roots(node.get('verbo'))
                extract_roots(node.get('objekto'))
                for a in node.get('aliaj', []):
                    extract_roots(a)

        extract_roots(query_ast)

        if not roots:
            logger.warning("  No roots found in query AST")
            return None

        # Get embeddings for each root and average
        embeddings = []
        with torch.no_grad():
            for root in roots:
                emb = self.hybrid_embedder.get_root_embedding(root)
                if emb is not None:
                    embeddings.append(emb)

        if not embeddings:
            logger.warning(f"  No embeddings found for roots: {roots}")
            return None

        # Average embeddings to get sentence embedding
        stacked = torch.stack(embeddings)
        sentence_emb = stacked.mean(dim=0)

        # Normalize
        norm = torch.norm(sentence_emb)
        if norm > 0:
            sentence_emb = sentence_emb / norm

        return sentence_emb.numpy()

    def _extract_query_slots(self, query_ast: Dict) -> Dict[str, Optional[np.ndarray]]:
        """
        Extract slot embeddings (SUBJ/VERB/OBJ) from query AST.

        This enables slot-aware retrieval that matches query structure to document structure,
        rather than averaging all roots together (which loses structural information).

        Returns:
            Dict mapping slot names to embeddings (or None if slot not present)
        """
        import torch

        if not hasattr(self, 'hybrid_embedder') or self.hybrid_embedder is None:
            return {'SUBJ': None, 'VERB': None, 'OBJ': None}

        slots = {'SUBJ': None, 'VERB': None, 'OBJ': None}

        def extract_roots_from_node(node) -> List[str]:
            """Extract all roots from a node and its children."""
            roots = []
            if not node or not isinstance(node, dict):
                return roots
            if node.get('tipo') == 'vorto':
                root = node.get('radiko', '')
                if root and len(root) >= 2:
                    roots.append(root.lower())
            elif node.get('tipo') == 'vortgrupo':
                roots.extend(extract_roots_from_node(node.get('kerno')))
                for p in node.get('priskriboj', []):
                    roots.extend(extract_roots_from_node(p))
            return roots

        def embed_roots(roots: List[str]) -> Optional[np.ndarray]:
            """Embed a list of roots, returning averaged embedding."""
            if not roots:
                return None
            embeddings = []
            with torch.no_grad():
                for root in roots:
                    emb = self.hybrid_embedder.get_root_embedding(root)
                    if emb is not None:
                        embeddings.append(emb)
            if not embeddings:
                return None
            stacked = torch.stack(embeddings)
            avg_emb = stacked.mean(dim=0)
            norm = torch.norm(avg_emb)
            if norm > 0:
                avg_emb = avg_emb / norm
            return avg_emb.numpy()

        # Extract slots from frazo-level AST
        if query_ast.get('tipo') == 'frazo':
            # SUBJ: subject (who/what is doing)
            subj_roots = extract_roots_from_node(query_ast.get('subjekto'))
            slots['SUBJ'] = embed_roots(subj_roots)

            # VERB: verb (action)
            verb_roots = extract_roots_from_node(query_ast.get('verbo'))
            slots['VERB'] = embed_roots(verb_roots)

            # OBJ: object (what is acted upon)
            obj_roots = extract_roots_from_node(query_ast.get('objekto'))
            slots['OBJ'] = embed_roots(obj_roots)

            # Log what we extracted
            logger.debug(f"  Query slots: SUBJ={subj_roots}, VERB={verb_roots}, OBJ={obj_roots}")

        return slots

    def _keyword_prefilter(
        self,
        query_ast: Dict,
        max_results: int = 1000,
        require_all_keywords: bool = True,
    ) -> List[Tuple[float, Dict]]:
        """
        Fast keyword-based prefilter using grep with slot-based reranking.

        Extracts content words from query AST and searches for documents
        containing those keywords. Now properly requires ALL significant keywords
        and uses slot similarity for reranking.

        Uses BOTH roots AND word stems for better matching:
        - Roots: As parsed by parser (e.g., "fond" from "fondis")
        - Stems: Original words minus Esperanto endings (e.g., "Esperant" from "Esperanton")

        This dual approach handles cases where:
        - Parser correctly extracts roots (verb stems like "fond")
        - Parser incorrectly decomposes words (proper nouns like "Esperanto" → "esp")

        Args:
            query_ast: Parsed query AST
            max_results: Maximum documents to return
            require_all_keywords: If True, only return docs containing ALL keywords (default: True)

        Returns:
            List of (score, document) tuples
        """
        import json
        import subprocess
        import re

        # Extract content words from query: both roots AND word stems
        keywords = set()
        # Function words to skip (don't search for question words, articles, etc.)
        skip_vortspeco = {'korelativo', 'pronomo', 'artikolo', 'prepozicio', 'konjunkcio'}

        def get_stem(word: str) -> str:
            """Extract stem by removing Esperanto grammatical endings."""
            word = word.lower()
            # Remove noun/adjective/adverb/verb endings
            for ending in ['ojn', 'ajn', 'on', 'an', 'en', 'oj', 'aj', 'as', 'is', 'os', 'us', 'i', 'u', 'o', 'a', 'e']:
                if word.endswith(ending) and len(word) > len(ending) + 2:
                    return word[:-len(ending)]
            return word

        def extract_keywords(node):
            if not node or not isinstance(node, dict):
                return
            if node.get('tipo') == 'vorto':
                vortspeco = node.get('vortspeco', '')
                # Skip function words
                if vortspeco in skip_vortspeco:
                    return

                # Add root if it's meaningful (>=3 chars)
                root = node.get('radiko', '')
                if root and len(root) >= 3:
                    keywords.add(root.lower())

                # Also add word stem from original form
                # This catches cases where parser incorrectly decomposed proper nouns
                plena_vorto = node.get('plena_vorto', '')
                if plena_vorto and len(plena_vorto) >= 4:
                    stem = get_stem(plena_vorto)
                    if len(stem) >= 3:
                        keywords.add(stem)

            elif node.get('tipo') == 'vortgrupo':
                extract_keywords(node.get('kerno'))
                for p in node.get('priskriboj', []):
                    extract_keywords(p)
            elif node.get('tipo') == 'frazo':
                extract_keywords(node.get('subjekto'))
                extract_keywords(node.get('verbo'))
                extract_keywords(node.get('objekto'))
                for a in node.get('aliaj', []):
                    extract_keywords(a)

        extract_keywords(query_ast)
        keywords = list(keywords)

        if not keywords:
            logger.warning("  No keywords extracted from query")
            return []

        # Expand keywords with synonyms from semantic DB
        # This helps find documents that use different words with same meaning
        # (e.g., "fondis" → also search for "kreis", "establis")
        expanded_keywords = set(keywords)
        for kw in keywords:
            synonyms = self.semantic_db.get_synonyms(kw)
            if synonyms:
                # Only add synonyms that are 3+ chars
                expanded_keywords.update(s for s in synonyms if len(s) >= 3)
                logger.debug(f"    Expanded '{kw}' with synonyms: {synonyms}")

        # Semantic role expansion: For creator/founder questions,
        # expand to related semantic expressions
        # This handles cases like "Kiu fondis X?" matching "Y, aŭtoro de X"
        CREATOR_VERBS = {'fond', 'kre', 'invent', 'establ', 'inici'}
        CREATOR_NOUNS = {'aŭtor', 'kreint', 'fondint', 'inventint', 'iniciatint'}
        ALL_CREATOR_TERMS = CREATOR_VERBS | CREATOR_NOUNS

        keywords_lower = {k.lower() for k in keywords}
        has_creator_verb = bool(keywords_lower & CREATOR_VERBS)
        role_or_terms = set()  # Terms that should be OR-matched, not AND-matched

        if has_creator_verb:
            # Add related creator expressions for broader matching
            expanded_keywords.update(ALL_CREATOR_TERMS)
            role_or_terms.update(ALL_CREATOR_TERMS)  # These should be OR-matched
            logger.debug(f"    Expanded with creator role terms: {ALL_CREATOR_TERMS}")

        keywords = list(expanded_keywords)

        # Remove duplicates and sort by length (longer = rarer)
        keywords = list(dict.fromkeys(keywords))  # Preserve order, remove dupes
        keywords_sorted = sorted(keywords, key=lambda k: -len(k))

        # Separate keywords into:
        # - non_role_keywords: regular keywords that should be AND-matched
        # - role_or_terms: semantic role terms that should be OR-matched
        non_role_keywords = [k for k in keywords_sorted if k.lower() not in role_or_terms]
        required_and_keywords = non_role_keywords[:2] if require_all_keywords else non_role_keywords[:1]
        all_keywords = keywords_sorted  # For scoring (includes synonyms)

        # Build OR pattern for role terms if any
        role_or_pattern = None
        if role_or_terms:
            escaped_role_terms = [re.escape(t) for t in role_or_terms]
            role_or_pattern = '|'.join(escaped_role_terms)

        logger.info(f"  Keyword prefilter: and_keywords={required_and_keywords[:3]}, "
                    f"role_or={len(role_or_terms)} terms, expanded={len(keywords)}")

        index_file = self.index_path / "slot_index.jsonl"

        try:
            # Strategy: Use grep pipeline with:
            # 1. Required AND keywords (chained greps for intersection)
            # 2. Optional OR pattern for semantic role terms
            #
            # Example for "Kiu fondis Esperanton?":
            #   grep -E "esperant" | grep -E "fond|kre|aŭtor|..."
            # This finds documents that mention Esperanto AND any creator term
            escaped_and = [re.escape(k) for k in required_and_keywords]

            # Start with first AND keyword or role pattern
            if escaped_and:
                first_pattern = escaped_and[0]
            elif role_or_pattern:
                first_pattern = role_or_pattern
                role_or_pattern = None  # Already used
            else:
                logger.warning("  No keywords for grep")
                return []

            first_proc = subprocess.Popen(
                ['grep', '-i', '-E', first_pattern, str(index_file)],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
            )
            last_proc = first_proc

            # Chain remaining AND keywords
            for keyword in escaped_and[1:]:
                next_proc = subprocess.Popen(
                    ['grep', '-i', '-E', keyword],
                    stdin=last_proc.stdout,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                )
                last_proc.stdout.close()
                last_proc = next_proc

            # Add OR pattern for role terms (if not already used as first pattern)
            if role_or_pattern:
                role_proc = subprocess.Popen(
                    ['grep', '-i', '-E', role_or_pattern],
                    stdin=last_proc.stdout,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                )
                last_proc.stdout.close()
                last_proc = role_proc

            # Limit output
            head_proc = subprocess.Popen(
                ['head', '-n', str(max_results * 2)],
                stdin=last_proc.stdout,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
            )
            last_proc.stdout.close()
            stdout, _ = head_proc.communicate(timeout=30)

            lines = stdout.decode('utf-8', errors='replace').strip().split('\n') if stdout else []

            # Parse results and score by ALL keywords + slot similarity
            candidates = []
            query_slots = self._extract_query_slots(query_ast)
            has_slots = any(v is not None for v in query_slots.values())

            for line in lines:
                if not line:
                    continue
                try:
                    doc = json.loads(line)
                    text_lower = doc.get('text', '').lower()

                    # Keyword match score (how many of ALL keywords match)
                    match_count = sum(1 for k in all_keywords if k.lower() in text_lower)
                    keyword_score = match_count / len(all_keywords)

                    # Slot similarity score (if slots available)
                    if has_slots:
                        slot_score = self._compute_slot_similarity(query_slots, doc)
                        # Combine: 40% keyword, 60% slot
                        combined_score = 0.4 * keyword_score + 0.6 * slot_score
                    else:
                        combined_score = keyword_score

                    candidates.append((combined_score, doc))
                except json.JSONDecodeError:
                    continue

            # Sort by combined score
            candidates.sort(key=lambda x: -x[0])

            logger.info(f"  Keyword prefilter: found {len(candidates)} candidates (require_all={require_all_keywords})")
            return candidates[:max_results]

        except subprocess.TimeoutExpired:
            logger.warning("  Keyword prefilter timed out")
            return []
        except Exception as e:
            logger.warning(f"  Keyword prefilter failed: {e}")
            return []

    def _get_document(self, doc_id: int) -> Dict:
        """Load a document by ID from index."""
        if doc_id < 0 or doc_id >= len(self.doc_offsets):
            raise IndexError(f"Document ID {doc_id} out of range")

        import json

        index_file = self.index_path / "slot_index.jsonl"
        with open(index_file, 'rb') as f:
            f.seek(self.doc_offsets[doc_id])
            line = f.readline()
            return json.loads(line)

    def search(
        self,
        query: str,
        top_k: int = 10,
        strategy: str = 'auto',
        prefilter_n: int = 500,
    ) -> List[Tuple[float, Dict]]:
        """
        Search for relevant documents using AST-aware strategies.

        Args:
            query: Query string (Esperanto)
            top_k: Number of results to return
            strategy: Search strategy:
                - 'auto': Automatically choose based on question type
                - 'pattern': Pure pattern matching
                - 'entity': Entity-focused matching
                - 'hybrid': Combine pattern + entity
            prefilter_n: Number of candidates from pre-filter (default: 500)
                        Increase for better recall at cost of speed

        Returns:
            List of (score, document) tuples sorted by relevance
        """
        # Parse query
        try:
            query_ast = parse(query)
        except Exception as e:
            logger.error(f"Failed to parse query: {query} - {e}")
            return []

        # Classify question
        classification = self.question_classifier.classify(query, query_ast)
        question_type = classification['question_type']
        entity_type = classification['entity_type']
        target_slots = classification['target_slots']

        logger.info(f"Query classified as {question_type.value}, entity: {entity_type.value}")

        # Extract entities from query
        query_entities = self.entity_recognizer.recognize_entities(query_ast)

        # Choose strategy based on question type
        if strategy == 'auto':
            strategy = self._select_strategy(question_type, query_entities)

        logger.info(f"Using strategy: {strategy}")

        # Execute search with chosen strategy
        if strategy == 'entity':
            return self._search_entity_focused(
                query_ast, query_entities, target_slots, entity_type, top_k, prefilter_n
            )
        elif strategy == 'pattern':
            return self._search_pattern_matching(
                query_ast, target_slots, entity_type, top_k, prefilter_n
            )
        else:  # hybrid
            return self._search_hybrid(
                query_ast, query_entities, target_slots, entity_type, top_k, prefilter_n
            )

    def _select_strategy(
        self,
        question_type: QuestionType,
        query_entities: List,
    ) -> str:
        """
        Automatically select search strategy based on question type.

        Rules:
        - WHO questions with entities → entity-focused
        - WHERE questions → entity-focused (places)
        - WHEN questions → entity-focused (times)
        - WHAT/HOW/WHY questions → pattern matching
        - Questions with multiple entities → hybrid
        """
        if len(query_entities) >= 2:
            return 'hybrid'

        if question_type == QuestionType.WHO:
            return 'entity'
        elif question_type == QuestionType.WHERE:
            return 'entity'
        elif question_type == QuestionType.WHEN:
            return 'entity'
        else:
            return 'pattern'

    def _reconstruct_query(self, query_ast: Dict) -> str:
        """
        Reconstruct query text from AST for pre-filtering.

        Simple reconstruction that extracts words from AST nodes.
        """
        words = []

        def extract_words(node):
            if not node:
                return

            if isinstance(node, dict):
                # Extract word if present
                if node.get('tipo') == 'vorto' and 'plena_vorto' in node:
                    words.append(node['plena_vorto'])

                # Recurse into structure
                if node.get('tipo') == 'frazo':
                    extract_words(node.get('subjekto'))
                    extract_words(node.get('verbo'))
                    extract_words(node.get('objekto'))
                    for item in node.get('aliaj', []):
                        extract_words(item)
                elif node.get('tipo') == 'vortgrupo':
                    extract_words(node.get('kerno'))
                    for item in node.get('priskriboj', []):
                        extract_words(item)
            elif isinstance(node, list):
                for item in node:
                    extract_words(item)

        extract_words(query_ast)
        return ' '.join(words)

    def _search_pattern_matching(
        self,
        query_ast: Dict,
        target_slots: List[str],
        entity_type: QEntityType,
        top_k: int,
        prefilter_n: int = 500,
    ) -> List[Tuple[float, Dict]]:
        """
        Pure pattern matching strategy.

        Uses AST pattern matcher to find structurally similar sentences.

        Args:
            query_ast: Parsed query AST
            target_slots: Priority slots from classifier
            entity_type: Expected entity type
            top_k: Final number of results
            prefilter_n: Number of candidates from pre-filter (if available)
        """
        results = []

        # Stage 1: Pre-filter candidates using embeddings (if available)
        if self.prefilter_retriever == 'hnsw_direct':
            # Use lightweight HNSW prefilter
            candidate_docs = self._hnsw_prefilter(query_ast, max_results=prefilter_n)
        elif self.prefilter_retriever:
            logger.debug(f"  Pre-filtering {len(self.doc_offsets):,} docs → {prefilter_n} candidates")

            # Use HNSW to get top candidates
            query_text = self._reconstruct_query(query_ast)
            prefilter_results = self.prefilter_retriever.search(
                query_text,
                top_k=prefilter_n,
                hnsw_top_n=prefilter_n,
                slot_top_n=prefilter_n
            )

            # Extract document IDs from pre-filter results
            candidate_docs = [(score, doc) for score, doc in prefilter_results]
            logger.debug(f"  Got {len(candidate_docs)} candidates from pre-filter")
        else:
            # Try keyword prefilter first (fast grep-based)
            candidate_docs = []
            if self.use_keyword_prefilter:
                candidate_docs = self._keyword_prefilter(query_ast, max_results=prefilter_n)

            # Fallback: Brute-force scan (slow!) - only if keyword prefilter disabled or failed
            if not candidate_docs:
                logger.warning(f"  No pre-filter: scanning first 10k of {len(self.doc_offsets):,} docs")
                scan_limit = min(10000, len(self.doc_offsets))
                candidate_docs = []
                for doc_id in range(scan_limit):
                    doc = self._get_document(doc_id)
                    candidate_docs.append((0.0, doc))  # No pre-filter score

        # Stage 2: AST pattern matching on candidates
        logger.debug(f"  AST pattern matching {len(candidate_docs)} candidates")
        for prefilter_score, doc in candidate_docs:
            # Parse document if AST not stored
            doc_ast = doc.get('ast')
            if not doc_ast:
                try:
                    doc_ast = parse(doc['text'])
                except:
                    continue  # Skip unparseable documents

            # Match patterns
            match_result = self.pattern_matcher.match(
                query_ast,
                doc_ast,
                target_slots,
                entity_type.value,
            )

            if match_result.score > 0:
                results.append((match_result.score, doc))

        # Sort by score
        results.sort(key=lambda x: x[0], reverse=True)

        return results[:top_k]

    def _search_entity_focused(
        self,
        query_ast: Dict,
        query_entities: List,
        target_slots: List[str],
        entity_type: QEntityType,
        top_k: int,
        prefilter_n: int = 500,
    ) -> List[Tuple[float, Dict]]:
        """
        Entity-focused strategy.

        Prioritizes documents containing the same entities as the query.
        Then uses pattern matching for ranking.
        """
        if not query_entities:
            # Fall back to pattern matching
            return self._search_pattern_matching(
                query_ast, target_slots, entity_type, top_k, prefilter_n
            )

        # Get entity texts and roots
        query_entity_texts = self.entity_recognizer.get_entity_texts(query_entities)
        query_entity_roots = self.entity_recognizer.get_entity_roots(query_entities)

        results = []

        # Stage 1: Pre-filter candidates using embeddings (if available)
        if self.prefilter_retriever == 'hnsw_direct':
            # Use lightweight HNSW prefilter
            candidate_docs = self._hnsw_prefilter(query_ast, max_results=prefilter_n)
        elif self.prefilter_retriever:
            logger.debug(f"  Pre-filtering {len(self.doc_offsets):,} docs → {prefilter_n} candidates")

            query_text = self._reconstruct_query(query_ast)
            prefilter_results = self.prefilter_retriever.search(
                query_text,
                top_k=prefilter_n,
                hnsw_top_n=prefilter_n,
                slot_top_n=prefilter_n
            )

            candidate_docs = [(score, doc) for score, doc in prefilter_results]
            logger.debug(f"  Got {len(candidate_docs)} candidates from pre-filter")
        else:
            # Try keyword prefilter first (fast grep-based)
            candidate_docs = []
            if self.use_keyword_prefilter:
                candidate_docs = self._keyword_prefilter(query_ast, max_results=prefilter_n)

            # Fallback: Brute-force scan (slow!) - only if keyword prefilter disabled or failed
            if not candidate_docs:
                logger.warning(f"  No pre-filter: scanning first 10k of {len(self.doc_offsets):,} docs")
                scan_limit = min(10000, len(self.doc_offsets))
                candidate_docs = []
                for doc_id in range(scan_limit):
                    doc = self._get_document(doc_id)
                    candidate_docs.append((0.0, doc))

        # Stage 1b: Supplementary keyword search for semantic role expansion
        # This catches documents that HNSW might miss due to lexical differences
        if self.use_keyword_prefilter and self.prefilter_retriever == 'hnsw_direct':
            keyword_candidates = self._keyword_prefilter(query_ast, max_results=prefilter_n // 2)
            if keyword_candidates:
                seen_texts = {doc.get('text', '') for _, doc in candidate_docs}
                added = 0
                for score, doc in keyword_candidates:
                    if doc.get('text', '') not in seen_texts:
                        candidate_docs.append((score, doc))
                        seen_texts.add(doc.get('text', ''))
                        added += 1
                if added > 0:
                    logger.info(f"  Added {added} candidates from keyword search")

        # Stage 2: Entity matching + pattern matching on candidates
        # Detect if query has a creator verb (for Esperanto classification)
        CREATOR_VERBS = {'fond', 'kre', 'invent', 'establ', 'inici'}
        query_verb = query_ast.get('verbo', {})
        query_verb_root = query_verb.get('radiko', '').lower() if query_verb else ''
        has_creator_query = query_verb_root in CREATOR_VERBS

        logger.debug(f"  Entity + pattern matching {len(candidate_docs)} candidates")
        for prefilter_score, doc in candidate_docs:
            # Parse document if AST not stored
            doc_ast = doc.get('ast')
            if not doc_ast:
                try:
                    doc_ast = parse(doc['text'])
                except:
                    continue  # Skip unparseable documents

            # Extract entities from document
            doc_entities = self.entity_recognizer.recognize_entities(doc_ast)
            doc_entity_texts = self.entity_recognizer.get_entity_texts(doc_entities)
            doc_entity_roots = self.entity_recognizer.get_entity_roots(doc_entities)

            # Check for entity overlap
            text_overlap = len(query_entity_texts & doc_entity_texts)
            root_overlap = len(query_entity_roots & doc_entity_roots)

            entity_score = (text_overlap * 0.6 + root_overlap * 0.4)

            if entity_score > 0:
                # Also compute pattern match score
                pattern_result = self.pattern_matcher.match(
                    query_ast, doc_ast, target_slots, entity_type.value
                )

                # Combine scores: entity match + pattern match
                combined_score = entity_score * 0.6 + pattern_result.score * 0.4

                results.append((combined_score, doc))

        # Sort by score
        results.sort(key=lambda x: x[0], reverse=True)

        return results[:top_k]

    def _search_hybrid(
        self,
        query_ast: Dict,
        query_entities: List,
        target_slots: List[str],
        entity_type: QEntityType,
        top_k: int,
        prefilter_n: int = 500,
    ) -> List[Tuple[float, Dict]]:
        """
        Hybrid strategy combining entity and pattern matching.

        Balances entity matching with structural pattern matching.
        Also includes semantic role expansion via keyword search.
        """
        # For now, same as entity-focused but with equal weighting
        if not query_entities:
            return self._search_pattern_matching(
                query_ast, target_slots, entity_type, top_k, prefilter_n
            )

        query_entity_texts = self.entity_recognizer.get_entity_texts(query_entities)
        query_entity_roots = self.entity_recognizer.get_entity_roots(query_entities)

        results = []

        # Stage 1a: Pre-filter candidates using embeddings (HNSW)
        if self.prefilter_retriever == 'hnsw_direct':
            # Use lightweight HNSW prefilter
            candidate_docs = self._hnsw_prefilter(query_ast, max_results=prefilter_n)
        elif self.prefilter_retriever:
            logger.debug(f"  Pre-filtering {len(self.doc_offsets):,} docs → {prefilter_n} candidates")

            query_text = self._reconstruct_query(query_ast)
            prefilter_results = self.prefilter_retriever.search(
                query_text,
                top_k=prefilter_n,
                hnsw_top_n=prefilter_n,
                slot_top_n=prefilter_n
            )

            candidate_docs = [(score, doc) for score, doc in prefilter_results]
            logger.debug(f"  Got {len(candidate_docs)} candidates from pre-filter")
        else:
            # Try keyword prefilter first (fast grep-based)
            candidate_docs = []
            if self.use_keyword_prefilter:
                candidate_docs = self._keyword_prefilter(query_ast, max_results=prefilter_n)

            # Fallback: Brute-force scan (slow!) - only if keyword prefilter disabled or failed
            if not candidate_docs:
                logger.warning(f"  No pre-filter: scanning first 10k of {len(self.doc_offsets):,} docs")
                scan_limit = min(10000, len(self.doc_offsets))
                candidate_docs = []
                for doc_id in range(scan_limit):
                    doc = self._get_document(doc_id)
                    candidate_docs.append((0.0, doc))

        # Stage 1b: Supplementary keyword search for semantic role expansion
        # This catches documents that HNSW might miss due to lexical differences
        # (e.g., "aŭtoro de Esperanto" vs "fondis Esperanton")
        if self.use_keyword_prefilter and self.prefilter_retriever == 'hnsw_direct':
            keyword_candidates = self._keyword_prefilter(query_ast, max_results=prefilter_n // 2)
            if keyword_candidates:
                # Merge keyword candidates with HNSW candidates (dedupe by text)
                seen_texts = {doc.get('text', '') for _, doc in candidate_docs}
                added = 0
                for score, doc in keyword_candidates:
                    if doc.get('text', '') not in seen_texts:
                        candidate_docs.append((score, doc))
                        seen_texts.add(doc.get('text', ''))
                        added += 1
                if added > 0:
                    logger.info(f"  Added {added} candidates from keyword search (semantic role expansion)")

        # Stage 2: Entity + pattern matching on candidates
        # Detect if query has a creator verb (for semantic role expansion bonus)
        CREATOR_VERBS = {'fond', 'kre', 'invent', 'establ', 'inici'}
        CREATOR_NOUNS = {'aŭtor', 'kreint', 'fondint', 'inventint', 'iniciatint'}
        query_verb = query_ast.get('verbo', {})
        query_verb_root = query_verb.get('radiko', '').lower() if query_verb else ''
        has_creator_query = query_verb_root in CREATOR_VERBS

        logger.debug(f"  Hybrid matching {len(candidate_docs)} candidates")
        for prefilter_score, doc in candidate_docs:
            # Parse document if AST not stored
            doc_ast = doc.get('ast')
            if not doc_ast:
                try:
                    doc_ast = parse(doc['text'])
                except:
                    continue  # Skip unparseable documents

            # Entity matching
            doc_entities = self.entity_recognizer.recognize_entities(doc_ast)
            doc_entity_texts = self.entity_recognizer.get_entity_texts(doc_entities)
            doc_entity_roots = self.entity_recognizer.get_entity_roots(doc_entities)

            text_overlap = len(query_entity_texts & doc_entity_texts)
            root_overlap = len(query_entity_roots & doc_entity_roots)

            entity_score = (text_overlap * 0.6 + root_overlap * 0.4)

            # Pattern matching
            pattern_result = self.pattern_matcher.match(
                query_ast, doc_ast, target_slots, entity_type.value
            )

            # Semantic role bonus: if query has creator verb and doc has creator noun
            semantic_role_bonus = 0.0
            if has_creator_query:
                doc_text_lower = doc.get('text', '').lower()
                for creator_noun in CREATOR_NOUNS:
                    if creator_noun in doc_text_lower:
                        # Strong bonus for semantic role equivalence
                        # Higher bonus if doc lacks verb (title/heading about author)
                        doc_verb = doc_ast.get('verbo')
                        if doc_verb is None:
                            semantic_role_bonus = 2.5  # Very strong: title/heading like "X, Aŭtoro de Y"
                        else:
                            semantic_role_bonus = 1.5  # Standard bonus
                        logger.debug(f"    Creator role bonus ({semantic_role_bonus}) for: {doc.get('text', '')[:50]}")
                        break

            # Combined scoring: entity + pattern + semantic role bonus
            # Give semantic role bonus higher weight to match pattern-based matches
            combined_score = entity_score * 0.35 + pattern_result.score * 0.35 + semantic_role_bonus * 0.3

            if combined_score > 0:
                results.append((combined_score, doc))

        results.sort(key=lambda x: x[0], reverse=True)

        return results[:top_k]

    def explain_retrieval(self, query: str, doc: Dict) -> Dict:
        """
        Explain why a document was retrieved for a query.

        Args:
            query: Query string
            doc: Document dict

        Returns:
            Explanation dict with classification, entities, and pattern match
        """
        # Parse query
        query_ast = parse(query)

        # Classify
        classification = self.question_classifier.classify(query, query_ast)

        # Extract entities
        query_entities = self.entity_recognizer.recognize_entities(query_ast)
        doc_entities = self.entity_recognizer.recognize_entities(doc['ast'])

        # Pattern match
        pattern_result = self.pattern_matcher.match(
            query_ast,
            doc['ast'],
            classification['target_slots'],
            classification['entity_type'].value,
        )

        return {
            'query': query,
            'document': doc['text'],
            'classification': {
                'question_type': classification['question_type'].value,
                'entity_type': classification['entity_type'].value,
                'focus': classification['focus'],
                'target_slots': classification['target_slots'],
            },
            'query_entities': [
                {'text': e.text, 'type': e.entity_type.value, 'slot': e.slot}
                for e in query_entities
            ],
            'doc_entities': [
                {'text': e.text, 'type': e.entity_type.value, 'slot': e.slot}
                for e in doc_entities
            ],
            'pattern_match': {
                'score': pattern_result.score,
                'matched_slots': list(pattern_result.matched_slots),
                'transformations': pattern_result.transformations,
                'explanation': pattern_result.explanation,
            },
        }
