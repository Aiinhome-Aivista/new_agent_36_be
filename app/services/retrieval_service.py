"""
Retrieval & RAG Orchestration Service Module for DevKnow.

Coordinates the end-to-end Retrieval-Augmented Generation (RAG) workflow:
1. Query Routing & Security Screening: Invokes QueryRouter to detect injection attacks,
   out-of-scope non-technical queries, and extract language filters.
2. Vector Retrieval: Queries ChromaDB vector store for candidate functions matching
   the user query above the configurable similarity threshold.
3. Grounding Verification & Desync Detection: Queries relational database (MySQL/SQLite)
   to ensure candidates are verified and still exist. Identifies database drift
   and prevents hallucinated attribution.
4. Static Tree-sitter Code Inspection: Computes deterministic compiler metrics
   (parameters, return types, cyclomatic complexity, syntax validity) on attached or
   retrieved code snippets.
5. Grounding & Attribution Computation: Calculates verified percentage split between
   organizational knowledge base and general LLM synthesis.
6. Multi-step Reasoning Disclosure: Assembles real-time "thinking steps" explaining
   the AI's internal decision pipeline to users in the chat UI.
"""

import time
import json
import logging
from pathlib import Path
from typing import Dict, Any, Generator, List, Optional

from app.config import Config
from app.services.mysql_service import db_service
from app.services.chroma_service import chroma_service
from app.services.hybrid_search_service import hybrid_search_service
from app.services.gemini_service import gemini_service
from app.services.code_parser_service import code_parser
from app.services.query_router import QueryRouter
from app.models.database_models import FunctionRecord, Conversation, ConversationMessage
from app.rag.prompts import (
    GREETING_PROMPT_TEMPLATE,
    KB_HIT_PROMPT_TEMPLATE,
    KB_MISS_PROMPT_TEMPLATE,
    CODE_IMPROVEMENT_PROMPT_TEMPLATE,
    CODE_CONVERSION_PROMPT_TEMPLATE,
    OUT_OF_SCOPE_REFUSAL_PROMPT_TEMPLATE,
    PROMPT_INJECTION_REFUSAL_PROMPT_TEMPLATE,
    sanitize_prompt_input
)

logger = logging.getLogger("DevKnow.RAG")


class RetrievalService:
    """
    Core RAG orchestration service linking semantic search, relational verification,
    AST static analysis, and prompt generation.
    """

    @staticmethod
    def get_conversation_history(session_id: str, limit: int = 10) -> str:
        """
        Retrieves the last N conversation turns for context-aware multi-turn dialogue.

        Args:
            session_id: Unique chat session UUID.
            limit: Maximum number of previous message pairs to retrieve.

        Returns:
            str: Formatted conversation history string or 'None'.
        """
        with db_service.get_session() as session:
            conv = session.query(Conversation).filter_by(session_id=session_id).first()
            if not conv:
                return "None"
            msgs = (
                session.query(ConversationMessage)
                .filter_by(conversation_id=conv.id)
                .order_by(ConversationMessage.created_at.desc())
                .limit(limit)
                .all()
            )
            if not msgs:
                return "None"
            msgs.reverse()
            formatted = []
            for m in msgs:
                formatted.append(f"{m.role.upper()}: {m.message[:300]}")
            return "\n".join(formatted)

    @classmethod
    def reformulate_conversational_query(cls, query: str, session_id: str) -> str:
        """
        Reformulates conversational follow-up questions (e.g. 'what about timeouts for that?')
        into standalone search queries using recent dialogue keywords from conversation history.
        """
        if not Config.ENABLE_QUERY_REFORMULATION or not session_id:
            return query

        history = cls.get_conversation_history(session_id, limit=3)
        if not history or history == "None" or len(history.strip()) < 15:
            return query

        query_lower = query.lower().strip()
        words = query_lower.split()
        follow_up_tokens = {"that", "this", "it", "these", "those", "same", "above", "also", "instead", "why", "how"}
        has_signal = (
            len(words) <= 5 or
            any(w in follow_up_tokens for w in words) or
            any(ph in query_lower for ph in ["what about", "how about", "and for", "and in", "why did", "difference between"])
        )

        if not has_signal:
            return query

        prompt = (
            f"You are a developer search query contextualizer. Given the recent conversation history and a follow-up user query, "
            f"rewrite the follow-up query as a standalone search query that includes the relevant technical topic/subject.\n\n"
            f"Conversation History:\n{history[-600:]}\n\n"
            f"Follow-up Query: {query}\n\n"
            f"Return ONLY the rewritten search query. No quotes, no markdown, no explanation."
        )
        try:
            rewritten = gemini_service.generate_text(prompt)
            if rewritten:
                clean_rewrite = rewritten.strip().strip("\"'")
                if len(clean_rewrite) >= len(query) and len(clean_rewrite) < 180:
                    logger.info(f"Conversational Query Reformulation: '{query}' -> '{clean_rewrite}'")
                    return clean_rewrite
        except Exception as e:
            logger.debug(f"Query reformulation notice: {e}")

        return query

    @classmethod
    def execute_rag(
        cls,
        query: str,
        session_id: str,
        selected_language: Optional[str] = None,
        attached_code: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Executes the end-to-end RAG pipeline for a user message.

        Workflow:
        1. Query Routing: Detects adversarial injection, out-of-scope queries, and language filters.
        2. Vector Search: Finds top-K candidate embeddings in ChromaDB.
        3. Database Grounding: Verifies candidate functions exist in MySQL with `verified` status.
           Enforces strict anti-drift logging if ChromaDB has orphaned vector IDs.
        4. Static Code Inspection: Runs Tree-sitter on attached or retrieved code for complexity & smells.
        5. Prompt Assembly: Selects specialized prompt template (Hit, Miss, Conversion, Refusal).
        6. Attribution & Reasoning: Computes grounding percentages and UI reasoning disclosure steps.

        Args:
            query: The user's natural language question or instruction.
            session_id: Active session UUID for multi-turn history.
            selected_language: Optional language filter selected in the chat UI.
            attached_code: Optional code snippet attached with the message.

        Returns:
            Dict containing:
            - `source`: 'kb_hit' | 'llm_fallback' | 'security_blocked' | 'guardrail_blocked'
            - `intent`: Classified query intent string.
            - `max_score`: Peak semantic similarity score (0.0 to 1.0).
            - `retrieved_functions`: List of verified function dictionaries.
            - `prompt`: Fully assembled prompt ready for LLM generation.
            - `thinking_steps`: List of reasoning explanation steps for the UI.
            - `grounding_stats`: Percentage attribution between KB and LLM.
            - `execution_metadata`: Detailed audit trail and latency metrics.
        """
        start_time = time.time()

        # Step 1: Query Routing & Security Inspection
        routing = QueryRouter.route_query(
            query,
            user_selected_language=selected_language,
            attached_code=attached_code
        )
        intent = routing["intent"]
        filter_lang = routing["filter_language"]
        target_lang = routing["target_language"]
        is_security_blocked = routing.get("security_flag", False) or intent == "prompt_injection"
        is_out_of_scope = intent == "out_of_scope"

        # Step 1b: Conversational Query Reformulation for follow-up turns
        effective_query = cls.reformulate_conversational_query(query, session_id)
        is_query_reformulated = (effective_query != query)
        child_dependencies: List[Dict[str, Any]] = []

        # Step 2: Vector Search in ChromaDB (Bypassed if prompt_injection or out_of_scope)
        retrieved_functions: List[Dict[str, Any]] = []
        context_functions_str = ""

        if is_security_blocked or is_out_of_scope:
            vector_matches = []
            max_score = 0.0
            is_kb_hit = False
        else:
            search_query = effective_query
            if attached_code:
                search_query += f"\nCode Context: {attached_code[:500]}"

            # Step 2: Enterprise Hybrid Retrieval (Dense Vector + Exact Symbol RRF)
            vector_matches = hybrid_search_service.search(
                query=search_query,
                top_k=Config.RETRIEVAL_TOP_K,
                language=filter_lang
            )

            max_score = vector_matches[0]["similarity"] if vector_matches else 0.0
            has_potential_match = len(vector_matches) > 0 and max_score >= Config.SIMILARITY_THRESHOLD
            missing_ids: List[int] = []
            drift_detected = False

            # Step 3: Fetch exact original code from MySQL if vector match >= threshold
            if has_potential_match:
                mysql_ids = [m["mysql_id"] for m in vector_matches if m.get("mysql_id")]
                with db_service.get_session() as session:
                    # STRICT GUARD: Only retrieve functions that have been human-verified!
                    records = session.query(FunctionRecord).filter(
                        FunctionRecord.id.in_(mysql_ids),
                        (FunctionRecord.verification_status == "verified") | (FunctionRecord.verification_status.is_(None))
                    ).all()
                    record_map = {r.id: r for r in records}

                    for vm in vector_matches:
                        r_id = vm.get("mysql_id")
                        if r_id in record_map:
                            rec = record_map[r_id]
                            fn_dict = rec.to_dict()
                            fn_dict["similarity"] = vm["similarity"]
                            fn_dict["match_type"] = vm.get("match_type", "vector_semantic")
                            fn_dict["rrf_score"] = vm.get("rrf_score", 0.0)
                            retrieved_functions.append(fn_dict)

                missing_ids = [m_id for m_id in mysql_ids if m_id not in record_map]
                drift_detected = len(missing_ids) > 0

                # Explicit Audit Logging for Database Drift / Orphaned ChromaDB Matches
                if not retrieved_functions:
                    logger.error(
                        f"[RAG AUDIT: COMPLETE DATABASE DRIFT] "
                        f"ChromaDB returned {len(vector_matches)} candidate vector(s) (IDs: {mysql_ids}, top_similarity: {round(max_score, 4)}), "
                        f"but MySQL returned EMPTY (0 records in 'functions' table). "
                        f"User Query: '{query[:150]}' | Session: '{session_id}'. "
                        f"Action: Denied KB grounding. Forcing source='llm_fallback' with 0% KB attribution. "
                        f"Remediation: Click 'Sync Search Index' or call POST /api/knowledge-base/reindex."
                    )
                elif missing_ids:
                    logger.warning(
                        f"[RAG AUDIT: PARTIAL DATABASE DRIFT] "
                        f"ChromaDB returned candidates {mysql_ids}, but IDs {missing_ids} were not found in MySQL. "
                        f"Retrieved {len(retrieved_functions)} verified record(s). "
                        f"User Query: '{query[:150]}' | Session: '{session_id}'. "
                        f"Action: Filtered out orphaned IDs {missing_ids} from prompt context."
                    )
                else:
                    logger.info(
                        f"[RAG AUDIT: GROUNDING VERIFIED] "
                        f"Successfully verified {len(retrieved_functions)} function(s) in MySQL for candidate IDs: {mysql_ids} "
                        f"(Max similarity: {round(max_score, 4)})."
                    )

                # Step 3b: Call-Graph Dependency Expansion (Token-Budgeted, Multi-Hop, Proximity-Boosted)
                cg_used_tokens = 0
                cg_max_depth_reached = 1
                if retrieved_functions and Config.ENABLE_CALL_GRAPH_EXPANSION:
                    token_budget = Config.CALL_GRAPH_TOKEN_BUDGET
                    max_depth = Config.CALL_GRAPH_MAX_DEPTH
                    stub_threshold = Config.CALL_GRAPH_STUB_THRESHOLD_LINES
                    max_children = Config.CALL_GRAPH_MAX_CHILDREN
                    proximity_boost = Config.CALL_GRAPH_PROXIMITY_BOOST

                    known_names = {f["function_name"] for f in retrieved_functions}
                    call_queue: List[tuple] = []

                    # Seed queue with Level 1 calls from retrieved primary functions
                    for f in retrieved_functions:
                        body = f.get("function_body", "")
                        lang = f.get("programming_language", "python")
                        f_path = f.get("file_path") or f.get("rule_version_name")
                        calls = code_parser.extract_invoked_calls(body, lang)
                        for c in calls:
                            if c not in known_names and c not in [item[0] for item in call_queue]:
                                call_queue.append((c, f_path, 1))

                    while call_queue and len(child_dependencies) < max_children and cg_used_tokens < token_budget:
                        target_name, caller_file, current_depth = call_queue.pop(0)
                        if target_name in known_names:
                            continue

                        cg_max_depth_reached = max(cg_max_depth_reached, current_depth)

                        with db_service.get_session() as session:
                            candidates = session.query(FunctionRecord).filter(
                                FunctionRecord.function_name == target_name,
                                (FunctionRecord.verification_status == "verified") | (FunctionRecord.verification_status.is_(None))
                            ).all()

                            if not candidates:
                                known_names.add(target_name)
                                continue

                            def get_record_file(r: FunctionRecord) -> str:
                                sf = getattr(r, "source_file", None)
                                sf_path = getattr(sf, "file_path", None) if sf else None
                                return sf_path or getattr(r, "rule_version_name", "") or ""

                            # Rank candidates by file/directory proximity
                            def score_proximity(rec: FunctionRecord) -> int:
                                rec_file = get_record_file(rec)
                                if not proximity_boost or not caller_file or not rec_file:
                                    return 10
                                caller_p = Path(caller_file).name.lower()
                                rec_p = Path(rec_file).name.lower()
                                if rec_p == caller_p:
                                    return 100  # Exact same file
                                caller_parent = Path(caller_file).parent.name.lower()
                                rec_parent = Path(rec_file).parent.name.lower()
                                if caller_parent and rec_parent and caller_parent == rec_parent:
                                    return 50   # Sibling file in same directory/package
                                return 10       # Global match

                            candidates.sort(key=lambda r: (score_proximity(r), len((r.function_body or "").split("\n"))), reverse=True)
                            best_rec = candidates[0]
                            best_file = get_record_file(best_rec)
                            known_names.add(target_name)

                            body_lines = (best_rec.function_body or "").strip().split("\n")
                            line_count = len(body_lines)
                            raw_tokens = max(1, len(best_rec.function_body or "") // 4)

                            is_stub = False
                            selected_code = best_rec.function_body or ""

                            # Stub condition:
                            # 1. Depth > 1 (grandchildren are compact stubs to conserve budget)
                            # 2. Or lines exceed threshold lines (e.g. > 50 lines)
                            # 3. Or full body exceeds remaining token budget
                            if current_depth > 1 or line_count > stub_threshold or (cg_used_tokens + raw_tokens > token_budget):
                                stub_code = code_parser.generate_skeleton_stub(
                                    code=best_rec.function_body or "",
                                    language=best_rec.programming_language,
                                    max_lines=stub_threshold,
                                    source_file=best_file
                                )
                                stub_tokens = max(1, len(stub_code) // 4)
                                if cg_used_tokens + stub_tokens <= token_budget:
                                    selected_code = stub_code
                                    is_stub = True
                                    cg_used_tokens += stub_tokens
                                else:
                                    break
                            else:
                                cg_used_tokens += raw_tokens

                            c_dict = best_rec.to_dict()
                            c_dict["file_path"] = best_file
                            c_dict["function_body"] = selected_code
                            c_dict["is_stub"] = is_stub
                            c_dict["depth"] = current_depth
                            c_dict["match_type"] = f"ast_call_hop_{current_depth}"
                            child_dependencies.append(c_dict)

                            # If current depth < max_depth and full body was included, enqueue grandchild calls
                            if current_depth < max_depth and not is_stub and len(child_dependencies) < max_children:
                                child_calls = code_parser.extract_invoked_calls(best_rec.function_body or "", best_rec.programming_language)
                                for cc in child_calls:
                                    if cc not in known_names and cc not in [item[0] for item in call_queue]:
                                        call_queue.append((cc, best_file, current_depth + 1))

                    if child_dependencies:
                        logger.info(
                            f"[CALL-GRAPH EXPANSION] Successfully pulled {len(child_dependencies)} dependency(ies) "
                            f"(Depth: {cg_max_depth_reached}, Budget: {cg_used_tokens}/{token_budget} tokens): "
                            f"{[c['function_name'] + (' [stub]' if c.get('is_stub') else '') for c in child_dependencies]}"
                        )

                # Build readable context for prompt ONLY if verified functions exist in database
                if retrieved_functions:
                    context_blocks = []
                    for i, f in enumerate(retrieved_functions, 1):
                        is_summary = f.get("knowledge_source") == "file_summary" or f.get("function_name", "").startswith(Config.FILE_SUMMARY_PREFIX)
                        if is_summary:
                            block = [
                                f"### Architectural Blueprint #{i}: {f['function_name']} (Language: {f['programming_language']}, Version: {f['version']})",
                                f"- Scope: Hierarchical File Summary & Class/Function Catalog",
                                f"- Module/File: {f.get('rule_version_name', 'N/A')}",
                                f"- Description: {f.get('description', 'N/A')}",
                                f"- File Structure & Architecture Overview:\n```{f['programming_language'].lower()}\n{f['function_body']}\n```"
                            ]
                        else:
                            parent_cls = f.get("class_name") or "Top-level / Global"
                            block = [
                                f"### Function #{i}: {f['function_name']} (Language: {f['programming_language']}, Version: {f['version']})",
                                f"- Module/Rule: {f.get('rule_version_name', 'N/A')}",
                                f"- Parent Class: {parent_cls}",
                                f"- Namespace: {f.get('namespace') or 'N/A'}",
                                f"- Description: {f.get('description', 'N/A')}",
                                f"- Signature: {f['function_header']}",
                                f"- Exact Original Implementation:\n```{f['programming_language'].lower()}\n{f['function_body']}\n```"
                            ]
                        context_blocks.append("\n".join(block))

                    # Append direct child call dependencies discovered via Tree-sitter
                    if child_dependencies:
                        dep_blocks = ["\n## Direct Child Call Dependencies (Automatically resolved via Tree-sitter AST):"]
                        for j, cd in enumerate(child_dependencies, 1):
                            stub_note = " [Skeleton Stub - Interface Contract]" if cd.get("is_stub") else ""
                            hop_note = f"Hop {cd.get('depth', 1)}"
                            loc_note = cd.get("file_path") or cd.get("rule_version_name") or "Local Module"
                            dep_blocks.append(
                                f"### Helper Dependency #{j}: {cd['function_name']}{stub_note} (Level: {hop_note}, Origin: {loc_note})\n"
                                f"- Signature: {cd['function_header']}\n"
                                f"- Purpose: {cd.get('description', 'Internal helper function called by parent logic')}\n"
                                f"- Implementation / Contract:\n```{cd['programming_language'].lower()}\n{cd['function_body']}\n```"
                            )
                        context_blocks.append("\n\n".join(dep_blocks))

                    context_functions_str = "\n\n".join(context_blocks)

            # CRITICAL: A true KB Hit ONLY exists if verified records were successfully fetched from MySQL!
            is_kb_hit = len(retrieved_functions) > 0
            if not is_kb_hit:
                max_score = 0.0

        # Step 4: Retrieve Conversation Memory & Static Tree-sitter Inspection
        history = cls.get_conversation_history(session_id, limit=Config.CONVERSATION_MEMORY_TURNS)
        safe_code = sanitize_prompt_input(attached_code) if attached_code else ""
        attached_code_section = ""
        if safe_code:
            try:
                ast_info = code_parser.analyze_code_quality(safe_code, selected_language)
                smells_str = "; ".join(ast_info["code_smells"]) if ast_info["code_smells"] else "None detected (Clean structure)"
                syntax_str = "Valid Syntax" if ast_info["is_valid_syntax"] else f"Irregular ({ast_info['syntax_error']})"
                params_str = ", ".join(ast_info["parameters"]) if ast_info["parameters"] else "None identified"
                ast_summary = (
                    f"### STATIC TREE-SITTER CODE ANALYSIS (DETERMINISTIC COMPILER METRICS):\n"
                    f"- Language: {ast_info['language']}\n"
                    f"- Identified Function: {ast_info['function_name']}\n"
                    f"- Parameters: {params_str}\n"
                    f"- Return Type: {ast_info['return_type'] or 'Unspecified'}\n"
                    f"- Cyclomatic Complexity: {ast_info['complexity']}\n"
                    f"- Syntax Check: {syntax_str}\n"
                    f"- Static Code Smells: {smells_str}\n"
                )
            except Exception:
                ast_summary = ""

            attached_code_section = (
                f"### ATTACHED CODE SNIPPET (UNTRUSTED USER DATA):\n"
                f"<attached_code>\n{safe_code}\n</attached_code>\n\n"
                f"{ast_summary}"
            )
        safe_query = sanitize_prompt_input(query)

        # Step 5: Choose Prompt Template
        if is_security_blocked:
            source = "security_blocked"
            prompt = PROMPT_INJECTION_REFUSAL_PROMPT_TEMPLATE.format(query=safe_query)
        elif is_out_of_scope:
            source = "guardrail_blocked"
            prompt = OUT_OF_SCOPE_REFUSAL_PROMPT_TEMPLATE.format(query=safe_query)
        elif is_kb_hit:
            source = "kb_hit"
            if intent == "code_improvement":
                prompt = CODE_IMPROVEMENT_PROMPT_TEMPLATE.format(
                    context_functions=context_functions_str,
                    query=safe_query,
                    attached_code_section=attached_code_section
                )
            elif intent == "code_conversion":
                prompt = CODE_CONVERSION_PROMPT_TEMPLATE.format(
                    context_functions=context_functions_str,
                    target_language=target_lang or "target language",
                    query=safe_query,
                    attached_code_section=attached_code_section
                )
            else:
                prompt = KB_HIT_PROMPT_TEMPLATE.format(
                    context_functions=context_functions_str,
                    query=safe_query,
                    attached_code_section=attached_code_section,
                    history=history
                )
        else:
            source = "llm_fallback"
            if intent == "greeting":
                prompt = GREETING_PROMPT_TEMPLATE.format(query=safe_query)
            else:
                prompt = KB_MISS_PROMPT_TEMPLATE.format(
                    query=safe_query,
                    attached_code_section=attached_code_section,
                    history=history
                )

        latency_ms = round((time.time() - start_time) * 1000, 2)

        # Step 6: Compute Grounding & RAG Attribution Statistics
        if is_security_blocked:
            kb_ratio = 0
            llm_ratio = 0
            attribution_label = "Security Guardrail"
        elif is_out_of_scope:
            kb_ratio = 0
            llm_ratio = 0
            attribution_label = "Domain Guardrail"
        elif is_kb_hit:
            raw_kb = int(round(max_score * 100))
            kb_ratio = max(Config.RAG_KB_RATIO_MIN, min(Config.RAG_KB_RATIO_MAX, raw_kb))
            llm_ratio = 100 - kb_ratio
            attribution_label = "Runbook Grounded"
        else:
            kb_ratio = 0
            llm_ratio = 100
            attribution_label = "General AI Knowledge"

        grounding_stats = {
            "kb_percentage": kb_ratio,
            "llm_percentage": llm_ratio,
            "attribution_label": attribution_label,
            "similarity_score": round(max_score, 4),
            "retrieved_count": len(retrieved_functions),
            "child_dependencies_count": len(child_dependencies),
            "call_graph_tokens_used": cg_used_tokens,
            "call_graph_depth": cg_max_depth_reached
        }

        # Step 7: Assemble Thinking Process for UI Reasoning Disclosure
        if is_security_blocked:
            thinking_steps = [
                {
                    "title": "Security Guardrail Inspection",
                    "detail": "Adversarial prompt injection signature detected (attempted system instruction override or prompt leakage)."
                },
                {
                    "title": "Zero-Trust Enforcement",
                    "detail": "DevKnow enterprise security policy triggered. Execution halted before database query or code execution."
                },
                {
                    "title": "Security Intervention",
                    "detail": "Quarantining input and generating enterprise security notice."
                },
                {
                    "title": "Grounding & RAG Attribution",
                    "detail": "RAG Attribution: 0% Knowledge Base • 0% Code Synthesis (Security policy intervention enforced)."
                }
            ]
        elif is_out_of_scope:
            thinking_steps = [
                {
                    "title": "Domain Guardrail Inspection",
                    "detail": "Analyzed query against organizational safety and domain boundary rules. Detected out-of-scope non-technical topic (politics, world leaders, or general world trivia)."
                },
                {
                    "title": "Enterprise Boundary Policy",
                    "detail": "DevKnow is restricted to software engineering, codebase analysis, and internal runbooks. Non-technical inquiries are strictly prohibited."
                },
                {
                    "title": "Guardrail Enforcement",
                    "detail": "Vector search bypassed. Activating domain refusal prompt to maintain assistant integrity."
                },
                {
                    "title": "Grounding & RAG Attribution",
                    "detail": "RAG Attribution: 0% Knowledge Base • 0% General AI (Domain boundary refusal enforced)."
                }
            ]
        else:
            thinking_steps = [
                {
                    "title": "Intent & Language Routing",
                    "detail": f"Classified intent as '{intent.replace('_', ' ')}'. Language filter: '{filter_lang or 'all'}'."
                }
            ]
            if is_query_reformulated:
                thinking_steps.append({
                    "title": "Conversational Query Reformulation",
                    "detail": f"Contextualized follow-up query using conversation history: '{query}' -> '{effective_query}'."
                })
            thinking_steps.extend([
                {
                    "title": "Hybrid RAG Retrieval (Dense Vector + Exact Symbol RRF)",
                    "detail": (
                        f"Executed 2-stage hybrid search (dense ChromaDB vector + exact symbol lookup). "
                        f"Found {len(vector_matches)} candidate(s) (top similarity: {round(max_score * 100, 1)}%, "
                        f"top match type: {vector_matches[0].get('match_type', 'semantic')})."
                        if vector_matches else "Executed hybrid search. No semantic or exact symbol matches found."
                    )
                },
                {
                    "title": "Knowledge Grounding Check",
                    "detail": (
                        f"Retrieved {len(retrieved_functions)} verified function(s)/blueprint(s) verbatim from database."
                        if is_kb_hit
                        else (
                            f"Hybrid retrieval returned {len(vector_matches)} candidate(s), but corresponding code records were missing in MySQL database (database drift). Grounding denied; safely routing to General AI."
                            if (has_potential_match and not retrieved_functions)
                            else f"No company function exceeded similarity threshold ({Config.SIMILARITY_THRESHOLD}). Activating general developer intelligence."
                        )
                    )
                }
            ])
            if child_dependencies:
                stubs_count = sum(1 for c in child_dependencies if c.get("is_stub"))
                stub_detail = f" ({stubs_count} skeleton stub(s))" if stubs_count > 0 else ""
                thinking_steps.append({
                    "title": "Call-Graph Dependency Resolution (Tree-sitter AST)",
                    "detail": (
                        f"Traversed AST call graph (Depth: {cg_max_depth_reached}, Budget: {cg_used_tokens}/{Config.CALL_GRAPH_TOKEN_BUDGET} tokens). "
                        f"Pulled {len(child_dependencies)} verified helper dependency(ies){stub_detail}: "
                        f"({', '.join([c['function_name'] + (' [stub]' if c.get('is_stub') else '') for c in child_dependencies])}) into prompt context."
                    )
                })
            thinking_steps.extend([
                {
                    "title": "Response Synthesis",
                    "detail": "Grounding response in verified organizational code and runbook logic." if is_kb_hit else ("Synthesizing polite conversational greeting." if intent == "greeting" else "Synthesizing solution using idiomatic programming practices.")
                },
                {
                    "title": "Grounding & RAG Attribution",
                    "detail": f"Verified RAG ratio: {kb_ratio}% Knowledge Base (Hybrid match: {round(max_score * 100, 1)}%) + {llm_ratio}% LLM synthesis & reasoning." if is_kb_hit else "Verified RAG ratio: 0% Knowledge Base (no company runbook met threshold) + 100% General LLM programming knowledge."
                }
            ])

        # Step 7b: Run Tree-sitter Code Inspection on retrieved or attached code
        code_to_inspect = attached_code
        if not code_to_inspect and retrieved_functions:
            fn0 = retrieved_functions[0]
            header = fn0.get("function_header", "").strip()
            body = fn0.get("function_body", "").strip()
            lang = fn0.get("programming_language", "").lower()
            if lang == "python":
                code_to_inspect = f"{header}\n    {body}"
            else:
                code_to_inspect = f"{header} {{\n    {body}\n}}"

        if code_to_inspect and not is_security_blocked and not is_out_of_scope:
            inspect_lang = selected_language or (retrieved_functions[0]["programming_language"] if retrieved_functions else None)
            try:
                ast_metrics = code_parser.analyze_code_quality(code_to_inspect, inspect_lang)
                engine_tag = ast_metrics["parser_engine"].replace("_", "-").upper()
                thinking_steps.insert(-1, {
                    "title": f"Tree-sitter Syntax Inspection ({engine_tag})",
                    "detail": (
                        f"Parsed '{ast_metrics['function_name']}' | Complexity: {ast_metrics['complexity']} | "
                        f"Syntax: {'Valid' if ast_metrics['is_valid_syntax'] else 'Irregular'} | "
                        f"Parameters: {len(ast_metrics['parameters'])}."
                    )
                })
            except Exception as ast_err:
                logger.debug(f"AST inspection notice: {ast_err}")

        # Observability structured log
        obs_log = {
            "query": query,
            "session_id": session_id,
            "intent": intent,
            "filter_language": filter_lang,
            "source": source,
            "max_similarity_score": max_score,
            "grounding_stats": grounding_stats,
            "retrieved_count": len(retrieved_functions),
            "retrieved_ids": [f["id"] for f in retrieved_functions],
            "database_verification": {
                "vector_candidate_count": len(vector_matches),
                "vector_candidate_ids": [m.get("mysql_id") for m in vector_matches],
                "mysql_matched_count": len(retrieved_functions),
                "missing_mysql_ids": missing_ids,
                "desync_detected": drift_detected,
                "status": "grounded_ok" if is_kb_hit else ("vector_matched_but_mysql_empty" if (has_potential_match and not retrieved_functions) else "clean_kb_miss")
            },
            "latency_ms": latency_ms
        }
        logger.info(f"RAG Execution: {json.dumps(obs_log)}")

        return {
            "source": source,
            "intent": intent,
            "max_score": max_score,
            "retrieved_functions": retrieved_functions,
            "child_dependencies": child_dependencies,
            "prompt": prompt,
            "thinking_steps": thinking_steps,
            "grounding_stats": grounding_stats,
            "execution_metadata": obs_log
        }
