import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from app.services.mysql_service import db_service
from app.services.chroma_service import chroma_service
from app.services.kb_growth_service import KBGrowthService
from app.models.database_models import FunctionRecord

def main():
    print("=" * 65)
    print("🧪 LIVE SYSTEM VERIFICATION: RELATIONAL DB & VECTOR DB SYNC")
    print("=" * 65)

    # ─────────────────────────────────────────────────────────────
    # TEST 1: Dual-Database Synchronization on ADD
    # ─────────────────────────────────────────────────────────────
    print("\n[TEST 1] Verifying ADD / SAVE operation:")
    add_result = KBGrowthService.save_llm_answer_to_kb(
        original_query="How to parse JWT tokens in Python?",
        llm_answer="""```python
def decode_jwt_payload(token: str) -> dict:
    import json, base64
    payload_part = token.split('.')[1]
    padded = payload_part + '=' * (-len(payload_part) % 4)
    return json.loads(base64.urlsafe_b64decode(padded))
```""",
        function_name="decode_jwt_payload",
        programming_language="python",
        version="1.0.0",
        description="Decode and inspect JWT payload without external libraries"
    )
    test_id = add_result["function_id"]
    print(f" -> Inserted into system. Assigned ID: {test_id}")

    # Check MySQL
    with db_service.get_session() as session:
        mysql_row = session.query(FunctionRecord).filter_by(id=test_id).first()
        assert mysql_row is not None, "Failed: Record missing in MySQL!"
        print(f" ✅ Found in MySQL:")
        print(f"    - Function Name: {mysql_row.function_name}")
        print(f"    - Language:      {mysql_row.programming_language}")
        print(f"    - Version:       {mysql_row.version}")
        print(f"    - Code Body:     {mysql_row.function_body[:60]}...")

    # Check ChromaDB
    chroma_doc = chroma_service._collection.get(ids=[f"fn_{test_id}"])
    assert len(chroma_doc["ids"]) == 1, "Failed: Vector missing in ChromaDB!"
    meta = chroma_doc["metadatas"][0]
    print(f" ✅ Found in ChromaDB:")
    print(f"    - Vector ID:     {chroma_doc['ids'][0]}")
    print(f"    - MySQL Ref ID:  {meta['mysql_id']}")
    print(f"    - Version:       {meta['version']}")
    print(f"    - Language:      {meta['programming_language']}")

    # ─────────────────────────────────────────────────────────────
    # TEST 2: Versioning (Append-only / Multiple Versions of Same Function)
    # ─────────────────────────────────────────────────────────────
    print("\n[TEST 2] Verifying VERSIONING (Appending v2.0.0 alongside v1.0.0):")
    add_v2_result = KBGrowthService.save_llm_answer_to_kb(
        original_query="How to parse JWT tokens in Python v2?",
        llm_answer="""```python
def decode_jwt_payload_v2(token: str) -> dict:
    import jwt  # Using official PyJWT library
    return jwt.decode(token, options={"verify_signature": False})
```""",
        function_name="decode_jwt_payload",
        programming_language="python",
        version="2.0.0",
        description="Decode JWT payload using standard PyJWT library"
    )
    test_id_v2 = add_v2_result["function_id"]
    print(f" -> Inserted v2.0.0. Assigned ID: {test_id_v2}")

    with db_service.get_session() as session:
        all_versions = session.query(FunctionRecord).filter_by(function_name="decode_jwt_payload").order_by(FunctionRecord.id.asc()).all()
        assert len(all_versions) >= 2, "Failed: Both versions should be preserved in MySQL!"
        print(f" ✅ MySQL preserves ALL versions without overwriting:")
        for v in all_versions:
            print(f"    - ID: {v.id}, Version: {v.version}, Code Header: {v.function_header}")

    # Verify both versions searchable in ChromaDB
    c_v1 = chroma_service._collection.get(ids=[f"fn_{test_id}"])
    c_v2 = chroma_service._collection.get(ids=[f"fn_{test_id_v2}"])
    assert len(c_v1["ids"]) == 1 and len(c_v2["ids"]) == 1, "Both versions must be indexed in ChromaDB!"
    print(f" ✅ ChromaDB indexes both versions independently:")
    print(f"    - Vector fn_{test_id}: version={c_v1['metadatas'][0]['version']}")
    print(f"    - Vector fn_{test_id_v2}: version={c_v2['metadatas'][0]['version']}")

    # ─────────────────────────────────────────────────────────────
    # TEST 3: Synchronized Deletion / Purge in BOTH Databases
    # ─────────────────────────────────────────────────────────────
    print("\n[TEST 3] Verifying SYNCHRONIZED PURGE (Both databases updated):")
    del_success = KBGrowthService.delete_llm_generated(test_id)
    assert del_success is True, "Delete operation failed!"
    print(f" -> Executed delete for ID {test_id}")

    # Check MySQL after delete
    with db_service.get_session() as session:
        check_mysql = session.query(FunctionRecord).filter_by(id=test_id).first()
        assert check_mysql is None, "Failed: Record still exists in MySQL!"
        print(f" ✅ Confirmed removed from MySQL.")

    # Check ChromaDB after delete
    check_chroma = chroma_service._collection.get(ids=[f"fn_{test_id}"])
    assert len(check_chroma["ids"]) == 0, "Failed: Vector still exists in ChromaDB!"
    print(f" ✅ Confirmed removed from ChromaDB.")

    # Clean up v2 as well
    KBGrowthService.delete_llm_generated(test_id_v2)
    print(f" -> Cleaned up test record v2.")

    print("\n" + "=" * 65)
    print("🏆 ALL INTEGRATION CHECKS VERIFIED 100% OPERATIONAL!")
    print("=" * 65)

if __name__ == "__main__":
    main()

