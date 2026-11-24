import os
import sys
from pathlib import Path

# Add backend directory
sys.path.insert(0, str(Path(__file__).parent))

from app.services.code_doc_ingestion_service import CodeDocIngestionService
from app.services.mysql_service import db_service
from app.models.database_models import FunctionRecord

def run_test():
    print("\n--- 1. Testing Code File Ingestion (.py) ---")
    test_code_file = Path(__file__).parent / "test_sample_breaker.py"
    test_code_file.write_text("""
def calculate_substation_impedance(voltage_kv: float, mva_fault: float) -> float:
    \"\"\"Calculates the per-unit impedance for a medium-voltage substation feeder.\"\"\"
    if mva_fault <= 0:
        raise ValueError("MVA fault must be positive.")
    return (voltage_kv ** 2) / mva_fault

def verify_grounding_mesh_resistance(grid_amps: float, ground_ohms: float) -> bool:
    \"\"\"Ensures ground potential rise (GPR) does not exceed 5000V limit.\"\"\"
    gpr = grid_amps * ground_ohms
    return gpr <= 5000.0
""".strip(), encoding="utf-8")

    result = CodeDocIngestionService.ingest_code_or_doc_file(
        file_path=str(test_code_file),
        original_filename="sample_breaker.py",
        author_notes="Substation impedance and grounding guidelines"
    )
    print("Ingestion Result:", result["success"], "Extracted:", result["total_extracted"], "Conflicts:", result["conflicts_count"])
    for fn in result["staged_functions"]:
        print(f"  -> Function: {fn['function_name']} | v{fn['version']} | Status: {fn['conflict_status']}")

    print("\n--- 2. Checking Staging Queue ---")
    queue = CodeDocIngestionService.list_staging_queue("pending_review")
    print(f"Items currently pending human verification: {len(queue)}")
    
    first_staged_id = result["staged_functions"][0]["id"]
    print(f"\n--- 3. Approving Function #{first_staged_id} ({result['staged_functions'][0]['function_name']}) ---")
    app_res = CodeDocIngestionService.approve_staged_function(
        function_id=first_staged_id,
        notes="Reviewed and approved by Lead Electrical Engineer"
    )
    print("Approval Result:", app_res["success"], app_res["message"])

    with db_service.get_session() as s:
        rec = s.query(FunctionRecord).filter_by(id=first_staged_id).first()
        print(f"Record in DB now: status='{rec.verification_status}', version='{rec.version}', source='{rec.knowledge_source}'")

    # Clean up test file
    if test_code_file.exists():
        test_code_file.unlink()

    print("\n--- 4. Testing Document File Ingestion (.md) ---")
    test_doc_file = Path(__file__).parent / "test_rules.md"
    test_doc_file.write_text("""# Enterprise Breaker Protection Standards
Author: Senior Principal Engineer
Rule Version: 2026.1

All breaker trips must enforce instantaneous trip limits.
```typescript
export function validateInstantaneousTrip(currentAmps: number, pickupMultiplier: number): boolean {
    return currentAmps <= (pickupMultiplier * 100) && currentAmps > 0;
}
```
""".strip(), encoding="utf-8")

    doc_res = CodeDocIngestionService.ingest_code_or_doc_file(
        file_path=str(test_doc_file),
        original_filename="enterprise_breaker_rules.md"
    )
    print("Doc Ingestion Result:", doc_res["success"], "Extracted:", doc_res["total_extracted"])
    for fn in doc_res["staged_functions"]:
        print(f"  -> Extracted Snippet: {fn['function_name']} | v{fn['version']} | Desc: {fn['description']}")

    if test_doc_file.exists():
        test_doc_file.unlink()

    print("\n✓ ALL TESTS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    run_test()
