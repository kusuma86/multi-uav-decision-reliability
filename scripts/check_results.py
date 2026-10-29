"""Check row counts and key scenario names in generated CSV outputs."""
from pathlib import Path
import sys
import pandas as pd
checks = [
 ("FIMCAP_Experiment_11_Results/experiment11_runs.csv", 2880),
 ("Decision_Reliability_Audit_Results/decision_reliability_runs.csv", 2880),
 ("Decision_Reliability_Matched_Oracle_Results/matched_oracle_runs.csv", 1440),
 ("UAV_Resource_Constraint_Audit_Results/resource_audit_runs.csv", 360),
]
errors = 0
for filename, expected in checks:
    path = Path(filename)
    if not path.is_file():
        print(f"MISSING {path}")
        errors += 1
        continue
    df = pd.read_csv(path)
    valid = len(df) == expected
    if 'scenario' in df.columns and (df['scenario'] == 'shift_overconfidence').any():
        print(f"STALE SCENARIO NAME {path}")
        valid = False
    print(f"{'PASS' if valid else 'FAIL'} {path}: {len(df)} rows; expected {expected}")
    errors += int(not valid)
sys.exit(1 if errors else 0)
