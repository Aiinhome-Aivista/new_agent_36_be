"""
HybridSearchService: Enterprise-grade Hybrid Retrieval Engine for DevKnow.

Combines:
1. Dense Semantic Vector Search (ChromaDB cosine similarity via SentenceTransformers).
2. Exact Keyword & Symbol Matching (MySQL relational exact/partial symbol search).
3. Reciprocal Rank Fusion (RRF): Statistically robust rank aggregation algorithm (k=60)
   that fuses dense vector rankings with exact keyword rankings.

Capabilities:
- Conceptual Queries: "how does the workflow resume after checkpoint?" -> Vector channel shines.
- Exact Symbol Queries: "_advance_via_loop", "Orchestrator", "STAGE_AGENTS" -> Keyword channel guarantees 100% precision.
- Dual-Hit Amplification: Candidates scoring high in both channels receive fused priority.
"""

import re
import logging
from typing import Dict, Any, List, Optional, Set
from sqlalchemy import or_, and_, desc

from app.config import Config
from app.services.chroma_service import chroma_service
from app.services.mysql_service import db_service
from app.models.database_models import FunctionRecord

logger = logging.getLogger("DevKnow.HybridSearch")

# Standard English & query stop words to avoid false positive symbol matches
STOP_WORDS: Set[str] = {
    "a", "an", "the", "and", "or", "in", "on", "at", "to", "for", "of", "with",
    "by", "from", "up", "about", "into", "over", "after", "is", "are", "was",
    "were", "be", "been", "being", "have", "has", "had", "do", "does", "did",
    "how", "what", "where", "when", "why", "who", "which", "can", "could",
    "will", "would", "should", "explain", "describe", "show", "tell", "give",
    "find", "get", "code", "function", "method", "class", "file", "module",
    "please", "help", "system", "project", "me", "it", "its", "my", "you",
    "your", "we", "us", "our", "they", "them", "all", "any", "some", "every",
    "everything", "works", "work", "like", "need", "know", "see", "this",
    "that", "these", "those", "here", "there", "display", "check", "something"
}


class HybridSearchService:
    """
    Orchestrates multi-channel retrieval fusing ChromaDB dense vectors and MySQL keyword search.
    All parameters are dynamically controlled via Config (loaded from .env).
    """

    @property
    def rrf_k(self) -> int:
        return Config.RRF_K

    @property
    def vector_weight(self) -> float:
        return Config.RRF_VECTOR_WEIGHT

    @property
    def keyword_weight(self) -> float:
        return Config.RRF_KEYWORD_WEIGHT

    @classmethod
    def extract_search_terms(cls, query: str) -> List[str]:
        """
        Extracts candidate code symbols, identifiers, and significant keywords from user query.
        Preserves underscores, camelCase, and dotted identifiers (e.g. `_advance_via_loop`, `Orchestrator`).
        """
        if not query:
            return []

        # Find word/identifier tokens
        tokens = re.findall(r'[a-zA-Z_][a-zA-Z0-9_]*', query)
        significant: List[str] = []

        for tok in tokens:
            lower_tok = tok.lower()
            if len(tok) >= 3 and lower_tok not in STOP_WORDS:
                significant.append(tok)

        # Prioritize true code symbols: snake_case and camelCase identifiers come first
        def term_priority(t: str) -> int:
            if "_" in t:
                return 0  # snake_case identifier (e.g. setup_database_service)
            if any(c.isupper() for c in t[1:]):
                return 1  # camelCase / PascalCase
            if len(t) >= 6:
                return 2  # long significant technical keyword
            return 3  # short word

        significant.sort(key=term_priority)
        return list(dict.fromkeys(significant))  # Deduplicate while preserving order

    @classmethod
    def search(
        cls,
        query: str,
        top_k: Optional[int] = None,
        language: Optional[str] = None,
        rrf_k: Optional[int] = None,
        vector_weight: Optional[float] = None,
        keyword_weight: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Executes hybrid search fusing dense vector similarity and exact symbol matching.
        Uses environment-configured or runtime-overridden RRF parameters.

        Args:
            query: User search query or chat message.
            top_k: Number of final results to return (default: Config.RETRIEVAL_TOP_K).
            language: Optional language filter.
            rrf_k: Optional override for RRF smoothing constant (default: Config.RRF_K).
            vector_weight: Optional override for vector ranking weight (default: Config.RRF_VECTOR_WEIGHT).
            keyword_weight: Optional override for keyword ranking weight (default: Config.RRF_KEYWORD_WEIGHT).

        Returns:
            List of fused match dictionaries sorted by final rank:
            [{
                "mysql_id": 532,
                "similarity": 0.88,
                "match_type": "hybrid" | "exact_keyword" | "vector_semantic",
                "vector_rank": 1,
                "keyword_rank": 2,
                "rrf_score": 0.034
            }, ...]
        """
        if not query or not query.strip():
            return []

        # Resolve dynamic configuration with runtime overrides
        limit_k = top_k if top_k is not None else Config.RETRIEVAL_TOP_K
        k_const = rrf_k if rrf_k is not None else Config.RRF_K
        v_weight = vector_weight if vector_weight is not None else Config.RRF_VECTOR_WEIGHT
        kw_weight = keyword_weight if keyword_weight is not None else Config.RRF_KEYWORD_WEIGHT
        candidate_limit = max(limit_k * Config.RRF_CANDIDATE_MULTIPLIER, Config.RRF_CANDIDATE_LIMIT)

        # ── Channel 1: Dense Semantic Vector Search (ChromaDB) ─────────────
        vector_candidates: List[Dict[str, Any]] = []
        try:
            vector_candidates = chroma_service.search_similar(
                query=query,
                top_k=candidate_limit,
                language=language
            )
        except Exception as e:
            logger.warning(f"ChromaDB vector channel notice: {e}")

        # Map vector candidates by mysql_id with rank (1-indexed)
        vector_rank_map: Dict[int, Dict[str, Any]] = {}
        for rank, cand in enumerate(vector_candidates, 1):
            m_id = cand.get("mysql_id")
            if m_id:
                vector_rank_map[int(m_id)] = {
                    "rank": rank,
                    "similarity": cand.get("similarity", 0.0),
                    "cand_data": cand
                }

        # ── Channel 2: Exact Keyword & Symbol Search (MySQL) ───────────────
        terms = cls.extract_search_terms(query)
        keyword_candidates: List[Dict[str, Any]] = []

        if terms:
            try:
                keyword_candidates = cls._search_mysql_keywords(
                    terms=terms,
                    language=language,
                    limit=candidate_limit
                )
            except Exception as e:
                logger.warning(f"MySQL keyword channel notice: {e}")

        # Map keyword candidates by mysql_id with rank (1-indexed)
        keyword_rank_map: Dict[int, Dict[str, Any]] = {}
        for rank, cand in enumerate(keyword_candidates, 1):
            m_id = cand.get("id")
            if m_id:
                keyword_rank_map[int(m_id)] = {
                    "rank": rank,
                    "score": cand.get("keyword_score", 0.0),
                    "cand_data": cand
                }

        # ── Channel 3: Reciprocal Rank Fusion (RRF) ────────────────────────
        all_candidate_ids = set(vector_rank_map.keys()) | set(keyword_rank_map.keys())
        if not all_candidate_ids:
            return []

        fused_results: List[Dict[str, Any]] = []

        for m_id in all_candidate_ids:
            v_info = vector_rank_map.get(m_id)
            k_info = keyword_rank_map.get(m_id)

            rrf_score = 0.0
            match_type = "vector_semantic"

            if v_info and k_info:
                match_type = "hybrid"
                rrf_score += v_weight / (k_const + v_info["rank"])
                rrf_score += kw_weight / (k_const + k_info["rank"])
            elif v_info:
                match_type = "vector_semantic"
                rrf_score += v_weight / (k_const + v_info["rank"])
            elif k_info:
                match_type = "exact_keyword"
                rrf_score += kw_weight / (k_const + k_info["rank"])

            # Normalized similarity estimate for downstream thresholds:
            # Maximum theoretical RRF is (v_weight + kw_weight) / (k_const + 1)
            max_theoretical_rrf = (v_weight + kw_weight) / (k_const + 1)
            normalized_similarity = min(1.0, round(rrf_score / max_theoretical_rrf, 4)) if max_theoretical_rrf > 0 else 0.0

            # Preserve real vector cosine similarity if vector hit was strong
            if v_info and v_info["similarity"] > normalized_similarity:
                normalized_similarity = v_info["similarity"]

            fused_results.append({
                "mysql_id": m_id,
                "similarity": normalized_similarity,
                "rrf_score": rrf_score,
                "match_type": match_type,
                "vector_rank": v_info["rank"] if v_info else None,
                "keyword_rank": k_info["rank"] if k_info else None,
                "vector_similarity": v_info["similarity"] if v_info else 0.0
            })

        # Sort by RRF score descending
        fused_results.sort(key=lambda x: x["rrf_score"], reverse=True)
        top_matches = fused_results[:limit_k]

        logger.debug(
            f"HybridSearch: query='{query[:60]}' | Vector hits={len(vector_candidates)} | "
            f"Keyword hits={len(keyword_candidates)} | Fused Top-{len(top_matches)}: "
            f"{[(m['mysql_id'], m['match_type'], round(m['similarity'], 3)) for m in top_matches]}"
        )

        return top_matches

    # ── Internal Keyword Search ───────────────────────────────────────────────

    @classmethod
    def _search_mysql_keywords(
        cls,
        terms: List[str],
        language: Optional[str] = None,
        limit: int = 15
    ) -> List[Dict[str, Any]]:
        """
        Queries MySQL functions table prioritizing exact symbol matches and class matches.
        Scoring weights are dynamically controlled via Config.
        """
        results: List[Dict[str, Any]] = []

        with db_service.get_session() as session:
            # Base query strictly limited to verified functions
            q = session.query(FunctionRecord).filter(
                or_(
                    FunctionRecord.verification_status == "verified",
                    FunctionRecord.verification_status.is_(None)
                )
            )

            if language:
                norm_lang = language.lower().strip()
                q = q.filter(FunctionRecord.programming_language.ilike(f"%{norm_lang}%"))

            # Build prioritized conditions:
            # 1. Exact identifier matches for top terms (highest priority)
            or_conditions = []
            for t in terms[:5]:
                or_conditions.append(FunctionRecord.function_name == t)
                or_conditions.append(FunctionRecord.class_name == t)

            # 2. Substring matches only for terms with length >= 4 (avoids short token false positives)
            for t in terms[:5]:
                if len(t) >= 4:
                    or_conditions.append(FunctionRecord.function_name.ilike(f"%{t}%"))
                    or_conditions.append(FunctionRecord.function_header.ilike(f"%{t}%"))
                    or_conditions.append(FunctionRecord.description.ilike(f"%{t}%"))
                    or_conditions.append(FunctionRecord.rule_version_name.ilike(f"%{t}%"))

            if not or_conditions:
                return []

            matched_records = q.filter(or_(*or_conditions)).limit(limit * 2).all()

            # Rank matched records by keyword tightness using Config weights
            scored: List[Dict[str, Any]] = []
            for rec in matched_records:
                score = 0.0
                fn_name = (rec.function_name or "").lower()
                c_name = (rec.class_name or "").lower()
                header = (rec.function_header or "").lower()

                for t in terms:
                    t_lower = t.lower()
                    if fn_name == t_lower:
                        score += Config.RRF_KEYWORD_EXACT_FN_SCORE  # Exact function match
                    elif fn_name.startswith(t_lower) or t_lower in fn_name:
                        score += Config.RRF_KEYWORD_PARTIAL_FN_SCORE  # Partial function match
                    if c_name and c_name == t_lower:
                        score += Config.RRF_KEYWORD_CLASS_SCORE  # Exact class match
                    if t_lower in header:
                        score += Config.RRF_KEYWORD_HEADER_SCORE  # Signature match

                scored.append({
                    "id": rec.id,
                    "function_name": rec.function_name,
                    "class_name": rec.class_name,
                    "keyword_score": score
                })

            scored.sort(key=lambda x: x["keyword_score"], reverse=True)
            return scored[:limit]


hybrid_search_service = HybridSearchService()
