import json
import uuid
import logging
from flask import Blueprint, request, Response, stream_with_context, jsonify

from app.services.mysql_service import db_service
from app.services.retrieval_service import RetrievalService
from app.services.gemini_service import gemini_service
from app.models.database_models import Conversation, ConversationMessage

chat_bp = Blueprint("chat", __name__)
logger = logging.getLogger("DevKnow.ChatRoutes")

@chat_bp.route("/chat", methods=["POST"])
def chat():
    """
    RAG Chat endpoint with Server-Sent Events (SSE) token-by-token streaming.
    Event sequence:
    1. 'metadata' event: source (kb_hit vs llm_fallback), score, intent, retrieved functions.
    2. 'token' events: text chunks streamed from Gemini.
    3. 'done' event: marks stream completion.
    """
    data = request.get_json() or {}
    query = data.get("query", "").strip()
    session_id = data.get("session_id", "").strip()
    selected_language = data.get("selected_language", None)
    attached_code = data.get("attached_code", None)

    # Input guardrail
    if not query:
        return jsonify({"success": False, "error": "Query cannot be empty"}), 400

    if len(query) > 4000:
        return jsonify({"success": False, "error": "Query exceeds maximum limit of 4000 characters"}), 400

    if not session_id:
        session_id = str(uuid.uuid4())

    # Ensure conversation exists
    with db_service.get_session() as session:
        conv = session.query(Conversation).filter_by(session_id=session_id).first()
        if not conv:
            title = query[:50] + ("..." if len(query) > 50 else "")
            conv = Conversation(session_id=session_id, title=title)
            session.add(conv)
            session.flush()

        conv_id = conv.id

        # Save user message
        user_msg = ConversationMessage(
            conversation_id=conv_id,
            role="user",
            message=query,
            attached_code=attached_code,
            answer_source="none"
        )
        session.add(user_msg)

    # Execute RAG Retrieval & Prompt Assembly
    rag_result = RetrievalService.execute_rag(
        query=query,
        session_id=session_id,
        selected_language=selected_language,
        attached_code=attached_code
    )

    prompt = rag_result["prompt"]
    source = rag_result["source"]
    intent = rag_result["intent"]
    max_score = rag_result["max_score"]
    retrieved_functions = rag_result["retrieved_functions"]

    def generate_stream():
        # 1. Emit metadata event with thinking / reasoning steps
        meta_payload = {
            "session_id": session_id,
            "source": source,
            "intent": intent,
            "similarity_score": max_score,
            "retrieved_count": len(retrieved_functions),
            "retrieved_functions": retrieved_functions,
            "thinking_steps": rag_result.get("thinking_steps", []),
            "latency_ms": rag_result.get("execution_metadata", {}).get("latency_ms", 0)
        }
        yield f"event: metadata\ndata: {json.dumps(meta_payload)}\n\n"

        # 2. Stream tokens from Gemini
        full_response_text = []
        try:
            for token in gemini_service.stream_generate(prompt):
                full_response_text.append(token)
                token_data = json.dumps({"token": token})
                yield f"event: token\ndata: {token_data}\n\n"
        except Exception as e:
            logger.error(f"Error during stream generation: {e}")
            err_data = json.dumps({"token": f"\n\n[Generation error: {str(e)}]"})
            yield f"event: token\ndata: {err_data}\n\n"

        complete_text = "".join(full_response_text)

        # 3. Save assistant message to conversation history
        try:
            with db_service.get_session() as session:
                assistant_msg = ConversationMessage(
                    conversation_id=conv_id,
                    role="assistant",
                    message=complete_text,
                    answer_source=source,
                    retrieved_function_ids=",".join(str(f["id"]) for f in retrieved_functions),
                    similarity_score=max_score,
                    query_intent=intent
                )
                session.add(assistant_msg)
        except Exception as ex:
            logger.error(f"Failed to record assistant message in DB: {ex}")

        # 4. Emit done event
        yield f"event: done\ndata: {json.dumps({'status': 'finished', 'session_id': session_id})}\n\n"

    return Response(
        stream_with_context(generate_stream()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive"
        }
    )

@chat_bp.route("/conversations", methods=["GET"])
def list_conversations():
    """Lists recent conversation sessions"""
    with db_service.get_session() as session:
        convs = session.query(Conversation).order_by(Conversation.updated_at.desc()).limit(20).all()
        return jsonify({
            "success": True,
            "conversations": [c.to_dict() for c in convs]
        })

@chat_bp.route("/conversations/<session_id>", methods=["GET"])
def get_conversation(session_id):
    """Retrieves all messages for a specific session"""
    with db_service.get_session() as session:
        conv = session.query(Conversation).filter_by(session_id=session_id).first()
        if not conv:
            return jsonify({"success": False, "error": "Conversation not found"}), 404

        messages = [m.to_dict() for m in conv.messages]
        return jsonify({
            "success": True,
            "conversation": conv.to_dict(),
            "messages": messages
        })

@chat_bp.route("/conversations/<session_id>", methods=["DELETE"])
def delete_conversation(session_id):
    """Deletes or clears a conversation session"""
    with db_service.get_session() as session:
        conv = session.query(Conversation).filter_by(session_id=session_id).first()
        if not conv:
            return jsonify({"success": False, "error": "Conversation not found"}), 404

        session.delete(conv)
        return jsonify({"success": True, "message": "Conversation deleted successfully."})

