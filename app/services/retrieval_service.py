import time
import json
import logging
from typing import Dict, Any, Generator, List, Optional

from app.config import Config
from app.services.mysql_service import db_service
from app.services.chroma_service import chroma_service
from app.services.gemini_service import gemini_service
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
    @staticmethod
    def get_conversation_history(session_id: str, limit: int = 10) -> str:
        """Retrieves last N turns of conversation history for the session"""
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
    def execute_rag(
        cls,
        query: str,
        session_id: str,
        selected_language: Optional[str] = None,
        attached_code: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Executes end-to-end RAG workflow:
        Routing -> Vector Search -> Threshold Check -> MySQL Fetch -> Prompt Prep
        Returns: {
            "source": "kb_hit" | "llm_fallback",
            "intent": str,
            "max_score": float,
            "retrieved_functions": list,
            "prompt": str,
            "execution_metadata": dict
        }
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

        # Step 2: Vector Search in ChromaDB (Bypassed if prompt_injection or out_of_scope)
        retrieved_functions = []
        context_functions_str = ""

        if is_security_blocked or is_out_of_scope:
            vector_matches = []
            max_score = 0.0
            is_kb_hit = False
        else:
            search_query = query
            if attached_code:
                search_query += f"\nCode Context: {attached_code[:500]}"

            vector_matches = chroma_service.search_similar(
                query=search_query,
                top_k=Config.RETRIEVAL_TOP_K,
                language=filter_lang
            )

            max_score = vector_matches[0]["similarity"] if vector_matches else 0.0
            is_kb_hit = len(vector_matches) > 0 and max_score >= Config.SIMILARITY_THRESHOLD

            # Step 3: Fetch exact original code from MySQL if KB Hit
            if is_kb_hit:
                mysql_ids = [m["mysql_id"] for m in vector_matches if m.get("mysql_id")]
                with db_service.get_session() as session:
                    records = session.query(FunctionRecord).filter(FunctionRecord.id.in_(mysql_ids)).all()
                    record_map = {r.id: r for r in records}

                    for vm in vector_matches:
                        r_id = vm.get("mysql_id")
                        if r_id in record_map:
                            rec = record_map[r_id]
                            fn_dict = rec.to_dict()
                            fn_dict["similarity"] = vm["similarity"]
                            retrieved_functions.append(fn_dict)

                # Build readable context for prompt
                context_blocks = []
                for i, f in enumerate(retrieved_functions, 1):
                    block = [
                        f"### Function #{i}: {f['function_name']} (Language: {f['programming_language']}, Version: {f['version']})",
                        f"- Module/Rule: {f.get('rule_version_name', 'N/A')}",
                        f"- Class/Namespace: {f.get('class_name', '')} {f.get('namespace', '')}".strip() or "- Class/Namespace: None",
                        f"- Description: {f.get('description', 'N/A')}",
                        f"- Signature: {f['function_header']}",
                        f"- Exact Original Implementation:\n```{f['programming_language'].lower()}\n{f['function_body']}\n```"
                    ]
                    context_blocks.append("\n".join(block))
                context_functions_str = "\n\n".join(context_blocks)

        # Step 4: Retrieve Conversation Memory
        history = cls.get_conversation_history(session_id, limit=Config.CONVERSATION_MEMORY_TURNS)
        safe_code = sanitize_prompt_input(attached_code) if attached_code else ""
        attached_code_section = f"### ATTACHED CODE SNIPPET (UNTRUSTED USER DATA):\n<attached_code>\n{safe_code}\n</attached_code>" if safe_code else ""
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

        # Step 6: Assemble Thinking Process for UI Reasoning Disclosure
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
                }
            ]
        else:
            thinking_steps = [
                {
                    "title": "Intent & Language Routing",
                    "detail": f"Classified intent as '{intent.replace('_', ' ')}'. Language filter: '{filter_lang or 'all'}'."
                },
                {
                    "title": "Runbook Semantic Search",
                    "detail": f"Searched organizational runbooks — found {len(vector_matches)} candidate(s) (highest match: {round(max_score * 100, 1)}%)." if vector_matches else "Searched organizational runbooks. No semantic matches found."
                },
                {
                    "title": "Knowledge Grounding Check",
                    "detail": f"Retrieved {len(retrieved_functions)} verified function(s) verbatim from database." if is_kb_hit else f"No company function exceeded similarity threshold ({Config.SIMILARITY_THRESHOLD}). Activating general developer intelligence."
                },
                {
                    "title": "Response Synthesis",
                    "detail": "Grounding response in verified organizational code and runbook logic." if is_kb_hit else ("Synthesizing polite conversational greeting." if intent == "greeting" else "Synthesizing solution using idiomatic programming practices.")
                }
            ]

        # Observability structured log
        obs_log = {
            "query": query,
            "session_id": session_id,
            "intent": intent,
            "filter_language": filter_lang,
            "source": source,
            "max_similarity_score": max_score,
            "retrieved_count": len(retrieved_functions),
            "retrieved_ids": [f["id"] for f in retrieved_functions],
            "latency_ms": latency_ms
        }
        logger.info(f"RAG Execution: {json.dumps(obs_log)}")

        return {
            "source": source,
            "intent": intent,
            "max_score": max_score,
            "retrieved_functions": retrieved_functions,
            "prompt": prompt,
            "thinking_steps": thinking_steps,
            "execution_metadata": obs_log
        }

