import os
import pandas as pd

data = [
    {
        "RuleVersionName": "After Parts Import",
        "RuleVersionDescription": "Automated post-import rule validation for electrical components",
        "Rule Version Script Header/Name": "bool IsRealBreakerPart(AtomicPart Part)",
        "Rule Version Script Body": 'return Part.GetAttrValue("DAconfigPartID", String.Empty).EndsWith("MAGNUMBREAKER") && !Part.ATTREXISTS("MagBkrStylenumber");',
        "Programming Language": "c#",
        "Framework": ".NET",
        "Version": "1.0.0",
        "Class Name": "BreakerRules",
        "Namespace": "Eaton.Electrical.Breakers",
        "Description": "Checks if the designated part represents a genuine physical Magnum circuit breaker rather than a placeholder.",
        "Additional Metadata": "Breakers, Magnum, PartsImport"
    },
    {
        "RuleVersionName": "After Parts Import",
        "RuleVersionDescription": "Automated post-import rule validation for electrical components",
        "Rule Version Script Header/Name": "bool IsProvisionalBreakerPart(AtomicPart Part)",
        "Rule Version Script Body": 'return Part.ATTREXISTS("MagBkrStylenumber") && Part.GetAttrValue("DAconfigPartID", String.Empty).EndsWith("MAGNUMBREAKER");',
        "Programming Language": "c#",
        "Framework": ".NET",
        "Version": "1.0.0",
        "Class Name": "BreakerRules",
        "Namespace": "Eaton.Electrical.Breakers",
        "Description": "Checks if the part is a provisional style number breaker waiting for final configuration approval.",
        "Additional Metadata": "Breakers, StyleNumber, Provisional"
    },
    {
        "RuleVersionName": "After Parts Import",
        "RuleVersionDescription": "Automated post-import rule validation for electrical components",
        "Rule Version Script Header/Name": "bool IsBreakerPart(AtomicPart Part)",
        "Rule Version Script Body": "return IsRealBreakerPart(Part) || IsProvisionalBreakerPart(Part);",
        "Programming Language": "c#",
        "Framework": ".NET",
        "Version": "1.0.0",
        "Class Name": "BreakerRules",
        "Namespace": "Eaton.Electrical.Breakers",
        "Description": "General evaluation function verifying whether the part is either a real or provisional breaker.",
        "Additional Metadata": "Breakers, CompositeEvaluation"
    },
    {
        "RuleVersionName": "After Parts Import",
        "RuleVersionDescription": "Automated post-import rule validation for electrical components",
        "Rule Version Script Header/Name": "bool HasSamePNAttributes(AtomicPart Part)",
        "Rule Version Script Body": 'List<string> PNAttributes = new List<string> { "PNMOUNT", "PNSCHEME", "PNTYPE", "PNZONE" };\nforeach(string p in PNAttributes) {\n    if(!Part.ATTREXISTS(p)) return false;\n}\nreturn true;',
        "Programming Language": "c#",
        "Framework": ".NET",
        "Version": "1.0.0",
        "Class Name": "PartValidation",
        "Namespace": "Eaton.Electrical.Parts",
        "Description": "Validates that all essential part number attributes (mount, scheme, type, zone) are present on the part.",
        "Additional Metadata": "Attributes, PNMOUNT, PNSCHEME"
    },
    {
        "RuleVersionName": "After Parts Import",
        "RuleVersionDescription": "Terminal block synchronization",
        "Rule Version Script Header/Name": "bool IsTerminalBlock(AtomicPart part)",
        "Rule Version Script Body": 'return part.GetAttrValue("PNSCHEME", string.Empty) == "MAGBKR" && part.GetAttrValue("PNTYPE", string.Empty).StartsWith("TB");',
        "Programming Language": "c#",
        "Framework": ".NET",
        "Version": "1.1.0",
        "Class Name": "TerminalRules",
        "Namespace": "Eaton.Electrical.Terminals",
        "Description": "Identifies if an imported part belongs to a terminal block category for MAGBKR wiring schemes.",
        "Additional Metadata": "Wiring, TerminalBlocks, MAGBKR"
    },
    {
        "RuleVersionName": "After Parts Import",
        "RuleVersionDescription": "Current transformer resolution rules",
        "Rule Version Script Header/Name": "bool IsNeutralSensor(AtomicPart part)",
        "Rule Version Script Body": 'return topLevelPart.FAMILYOREMPTYSTRING == "Current Transformer" && topLevelPart.CHECKSTR("PNTYPE", "NEUTRAL");',
        "Programming Language": "c#",
        "Framework": ".NET",
        "Version": "1.0.0",
        "Class Name": "TransformerRules",
        "Namespace": "Eaton.Electrical.Transformers",
        "Description": "Determines if an atomic transformer part is mapped as a dedicated neutral sensor.",
        "Additional Metadata": "CurrentTransformer, NeutralSensor"
    },
    {
        "RuleVersionName": "back_up",
        "RuleVersionDescription": "Database backup and archive routines",
        "Rule Version Script Header/Name": "def backup_functions_table(db_connection, target_path: str) -> bool:",
        "Rule Version Script Body": 'import shutil\nimport datetime\ntimestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")\nbackup_file = f"{target_path}/db_backup_{timestamp}.sql"\nprint(f"Exporting database tables to {backup_file}")\nreturn True',
        "Programming Language": "python",
        "Framework": "StandardLib",
        "Version": "2.0.0",
        "Class Name": "DatabaseBackup",
        "Namespace": "core.maintenance.backup",
        "Description": "Generates a snapshot SQL dump of all function records and metadata for disaster recovery.",
        "Additional Metadata": "Backup, DisasterRecovery, Database"
    },
    {
        "RuleVersionName": "Utility",
        "RuleVersionDescription": "Common email format validation",
        "Rule Version Script Header/Name": "def validate_email_address(email: str) -> bool:",
        "Rule Version Script Body": "import re\nregex = r'^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\\.[a-zA-Z0-9-.]+$'\nreturn bool(re.match(regex, email.strip())) if email else False",
        "Programming Language": "python",
        "Framework": "StandardLib",
        "Version": "1.0.0",
        "Class Name": "ValidationUtils",
        "Namespace": "core.utils.validation",
        "Description": "Validates email address structure according to RFC specifications using compiled regular expressions.",
        "Additional Metadata": "Validation, Email, Regex"
    }
]

df = pd.DataFrame(data)
output_path = os.path.join(os.path.dirname(__file__), "sample_rulebook.xlsx")
df.to_excel(output_path, index=False)
print(f"Generated sample Excel file with {len(df)} records at: {output_path}")

