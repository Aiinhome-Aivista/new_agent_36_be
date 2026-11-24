"""
Quick smoke test for AST logging.
Run from: new_agent_36_be/
  python test_ast_log.py
"""
import sys, os, logging
# Directly import the module without going through app/__init__.py
sys.path.insert(0, os.path.dirname(__file__))
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(name)s] %(message)s")

# Direct import – bypass Flask app bootstrap
import importlib.util, pathlib

spec = importlib.util.spec_from_file_location(
    "code_parser_service",
    pathlib.Path(__file__).parent / "app" / "services" / "code_parser_service.py"
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
code_parser = mod.code_parser

# ── Test 1: Python function with ZeroDivisionError edge case ──────────────────
py_code = """
def calculate_inverse_time_trip(current: float, pickup: float, time_dial: float = 1.0) -> float:
    ratio = current / pickup
    denominator = (ratio ** 0.02) - 1.0
    return time_dial * (0.14 / denominator)
"""
r1 = code_parser.analyze_code_quality(py_code.strip(), "python")
print("[TEST 1 - Python AST]", r1)

# ── Test 2: TypeScript type guard – null safety check ─────────────────────────
ts_code = """export function isBreakerPart(part: Part): boolean {
  return part.family === "Circuit Breaker" || part.category === "BREAKER";
}"""
r2 = code_parser.analyze_code_quality(ts_code, "typescript")
print("[TEST 2 - TypeScript Tree-sitter]", r2)

# ── Test 3: JavaScript breaker amperage match ─────────────────────────────────
js_code = """function validateBreakerAmperageMatch(breakerAmps, busAmps) {
  return breakerAmps <= busAmps;
}"""
r3 = code_parser.analyze_code_quality(js_code, "javascript")
print("[TEST 3 - JavaScript Tree-sitter]", r3)

# ── Test 4: C# with high complexity ──────────────────────────────────────────
cs_code = """public bool VerifyAuxiliarySwitchContacts(Part part, int expectedFormC) {
    if (part == null) throw new ArgumentNullException(nameof(part));
    var contacts = part.GetAttrValue("AUX_CONTACTS_COUNT", 0);
    if (contacts < 0) return false;
    if (contacts == 0 && expectedFormC > 0) return false;
    return contacts >= expectedFormC;
}"""
r4 = code_parser.analyze_code_quality(cs_code, "c#")
print("[TEST 4 - C# Tree-sitter]", r4)

# ── Test 5: extract_all_functions from a whole Python file snippet ────────────
multi_code = """
def validate_breaker(amps, bus):
    if amps <= 0:
        return False
    return amps <= bus

async def fetch_part_data(part_id: int) -> dict:
    result = await db.get(part_id)
    return result

def calculate_power(voltage, current):
    return voltage * current
"""
funcs = code_parser.extract_all_functions(multi_code.strip(), "python")
print(f"[TEST 5 - extract_all_functions] Found {len(funcs)} functions:")
for f in funcs:
    print(f"   → {f['function_name']} | lines {f['start_line']}-{f['end_line']} | complexity {f['complexity']}")

print("\n✓ Done. Check: new_agent_36_be/data/logs/ast/ast_analysis.log")
