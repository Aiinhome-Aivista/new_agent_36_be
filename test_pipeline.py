import os
import sys

# Ensure backend directory is in sys.path
sys.path.insert(0, os.path.dirname(__file__))

from app.config import Config
from app.services.mysql_service import db_service
from app.services.excel_ingestion_service import ExcelIngestionService
from app.services.chroma_service import chroma_service
from app.services.retrieval_service import RetrievalService
from app.services.kb_growth_service import KBGrowthService
from app.models.database_models import FunctionRecord

def main():
    print("=" * 60)
    print("🧪 RUNNING DEVKNOW RAG & DATABASE PIPELINE VALIDATION TEST")
    print("=" * 60)

    # 1. Ingest sample rulebook
    sample_file = os.path.join(os.path.dirname(__file__), "sample_rulebook.xlsx")
    print(f"\n[1] Testing Excel Ingestion with: {sample_file}")
    summary = ExcelIngestionService.ingest_excel_file(sample_file, "sample_rulebook.xlsx")
    print(f"Ingestion Result: Success={summary.get('success')}")
    print(f"Total: {summary.get('total_records')}, Successful: {summary.get('successful_records')}, Languages: {summary.get('languages_detected')}")
    assert summary["success"] is True, "Excel ingestion failed!"
    assert summary["successful_records"] > 0, "No records ingested!"

    # 2. Verify MySQL persistence
    with db_service.get_session() as session:
        funcs = session.query(FunctionRecord).all()
        print(f"\n[2] Verified MySQL Source of Truth: {len(funcs)} functions stored.")
        for f in funcs[:3]:
            print(f"  - [{f.programming_language.upper()}] {f.function_name} (Module: {f.rule_version_name})")

    # 3. Test ChromaDB Semantic Search
    query = "Find functions that check if a part is a real breaker"
    print(f"\n[3] Testing ChromaDB Vector Retrieval for query: '{query}'")
    matches = chroma_service.search_similar(query, top_k=3, language="c#")
    print(f"ChromaDB returned {len(matches)} matches:")
    for m in matches:
        print(f"  - Match: {m['function_name']} (Language: {m['programming_language']}, Similarity: {m['similarity']})")
    assert len(matches) > 0, "ChromaDB returned no matches!"

    # 4. Test RAG Pipeline Execution
    print(f"\n[4] Testing End-to-End RAG Execution (KB Hit Verification)")
    rag_result = RetrievalService.execute_rag(
        query=query,
        session_id="test_session_1",
        selected_language="c#"
    )
    print(f"RAG Result Source: {rag_result['source']}")
    print(f"RAG Max Score: {rag_result['max_score']}")
    print(f"Intent Classified: {rag_result['intent']}")
    print(f"Retrieved Exact Functions count: {len(rag_result['retrieved_functions'])}")
    assert rag_result["source"] == "kb_hit", f"Expected kb_hit but got {rag_result['source']}"

    # 5. Test KB Miss & Self-Growing Knowledge Base Loop
    miss_query = "How to implement an LRU cache with expiration in Python?"
    print(f"\n[5] Testing KB Miss and Self-Growing Store-Back Loop")
    miss_result = RetrievalService.execute_rag(
        query=miss_query,
        session_id="test_session_1",
        selected_language="python"
    )
    print(f"Miss Query Result Source: {miss_result['source']} (Expected: llm_fallback)")
    assert miss_result["source"] == "llm_fallback", "Expected llm_fallback on unseen question!"

    # Simulate user clicking "Save to Knowledge Base"
    print("\n[6] Simulating developer clicking 'Save to Knowledge Base' for this new solution...")
    save_result = KBGrowthService.save_llm_answer_to_kb(
        original_query=miss_query,
        llm_answer="""```python
from collections import OrderedDict
import time

class ExpiringLRUCache:
    def __init__(self, capacity: int, ttl_seconds: int):
        self.capacity = capacity
        self.ttl = ttl_seconds
        self.cache = OrderedDict()

    def get(self, key):
        if key not in self.cache:
            return None
        val, expiry = self.cache[key]
        if time.time() > expiry:
            del self.cache[key]
            return None
        self.cache.move_to_end(key)
        return val

    def put(self, key, value):
        expiry = time.time() + self.ttl
        self.cache[key] = (value, expiry)
        self.cache.move_to_end(key)
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)
```""",
        function_name="ExpiringLRUCache",
        programming_language="python",
        framework="StandardLib",
        description="LRU Cache with TTL expiration policy in Python"
    )
    print(f"Saved into KB: Function ID {save_result['function_id']}, Name: {save_result['function_name']}")

    # Now ask the SAME question again: it MUST now be a KB Hit!
    print("\n[7] Querying again with same topic to verify Knowledge Base Growth:")
    re_result = RetrievalService.execute_rag(
        query=miss_query,
        session_id="test_session_1",
        selected_language="python"
    )
    print(f"Subsequent Query Result Source: {re_result['source']}")
    print(f"New Max Similarity Score: {re_result['max_score']}")
    print(f"Retrieved Learned Function: {[f['function_name'] for f in re_result['retrieved_functions']]}")
    assert re_result["source"] == "kb_hit", "Knowledge Base should have hit the newly learned function!"

    print("\n" + "=" * 60)
    print("🎉 ALL TESTS PASSED! RAG PIPELINE & SELF-GROWING KB VERIFIED!")
    print("=" * 60)

if __name__ == "__main__":
    main()

