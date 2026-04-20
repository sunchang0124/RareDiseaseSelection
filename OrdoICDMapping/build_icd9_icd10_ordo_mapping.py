#!/usr/bin/env python3
"""Build a final ORDO + ICD10 + ICD9 mapping table.

Inputs:
- icd10cmtoicd9gem.csv
- ordo_icd10_mapping.csv

Output:
- ordo_icd10_icd9_mapping.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


OUTPUT_FIELDS = [
    "ordo_id",
    "orpha_code",
    "ordo_label",
    "icd10_code",
    "icd10_code_nodot",
    "icd10_family4",
    "icd9_code",
    "icd9_code_nodot",
    "icd9_family4",
    "mapping_relation",
    "mapping_icd_relation",
    "mapping_validation_status",
    "gem_flags",
    "gem_approximate",
    "gem_no_map",
    "gem_combination",
    "gem_scenario",
    "gem_choice_list",
]


def normalize_code(code: str) -> str:
    return code.strip().upper().replace(".", "").replace(" ", "")


def icd9_family4(code: str) -> str:
    code = normalize_code(code)
    return code[:4]


def icd10_family4(code: str) -> str:
    code = normalize_code(code)
    return code[:4]


def load_gem_rows(path: Path):
    rows_by_icd10 = {}
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            icd10cm_code = row.get("icd10cm", "").strip()
            key = normalize_code(icd10cm_code)
            if not key:
                continue
            rows_by_icd10.setdefault(key, []).append(row)
    return rows_by_icd10


def load_ordo_rows(path: Path):
    rows_by_icd10 = {}
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            key = row.get("reference_nodot") or normalize_code(row.get("reference", ""))
            if not key:
                continue
            rows_by_icd10.setdefault(key, []).append(row)
    return rows_by_icd10


def main():
    parser = argparse.ArgumentParser(description="Join ORDO ICD10 mapping to ICD9 GEM")
    parser.add_argument(
        "--gem",
        default="icd10cmtoicd9gem.csv",
        help="Path to ICD10-CM to ICD9 GEM CSV",
    )
    parser.add_argument(
        "--ordo",
        default="ordo_icd10_mapping.csv",
        help="Path to ORDO ICD10 mapping CSV",
    )
    parser.add_argument(
        "--output",
        default="ordo_icd10_icd9_mapping.csv",
        help="Output CSV path",
    )
    args = parser.parse_args()

    gem_path = Path(args.gem)
    ordo_path = Path(args.ordo)
    output_path = Path(args.output)

    gem_by_icd10 = load_gem_rows(gem_path)
    ordo_by_icd10 = load_ordo_rows(ordo_path)

    with output_path.open("w", encoding="utf-8", newline="") as out_fh:
        writer = csv.DictWriter(out_fh, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()

        for icd10_code_nodot, ordo_rows in ordo_by_icd10.items():
            gem_rows = list(gem_by_icd10.get(icd10_code_nodot, []))
            if not gem_rows:
                # ORDO uses less granular ICD-10 codes, while GEM uses ICD-10-CM.
                # Match all ICD-10-CM descendants that share the undotted prefix.
                for gem_icd10_nodot, matched_rows in gem_by_icd10.items():
                    if gem_icd10_nodot.startswith(icd10_code_nodot):
                        gem_rows.extend(matched_rows)
            for ordo_row in ordo_rows:
                if gem_rows:
                    for gem_row in gem_rows:
                        icd9_code = gem_row.get("icd9cm", "").strip()
                        icd10_code = gem_row.get("icd10cm", "").strip() or ordo_row.get("reference", "")
                        writer.writerow({
                            "ordo_id": ordo_row.get("ordo_id", ""),
                            "orpha_code": ordo_row.get("orpha_code", ""),
                            "ordo_label": ordo_row.get("disorder_name", ""),
                            "icd10_code": icd10_code,
                            "icd10_code_nodot": icd10_code_nodot,
                            "icd10_family4": icd10_family4(icd10_code),
                            "icd9_code": icd9_code,
                            "icd9_code_nodot": normalize_code(icd9_code),
                            "icd9_family4": icd9_family4(icd9_code),
                            "mapping_relation": ordo_row.get("mapping_relation", ""),
                            "mapping_icd_relation": ordo_row.get("mapping_icd_relation", ""),
                            "mapping_validation_status": ordo_row.get("mapping_validation_status", ""),
                            "gem_flags": gem_row.get("flags", ""),
                            "gem_approximate": gem_row.get("approximate", ""),
                            "gem_no_map": gem_row.get("no_map", ""),
                            "gem_combination": gem_row.get("combination", ""),
                            "gem_scenario": gem_row.get("scenario", ""),
                            "gem_choice_list": gem_row.get("choice_list", ""),
                        })
                else:
                    writer.writerow({
                        "ordo_id": ordo_row.get("ordo_id", ""),
                        "orpha_code": ordo_row.get("orpha_code", ""),
                        "ordo_label": ordo_row.get("disorder_name", ""),
                        "icd10_code": ordo_row.get("reference", ""),
                        "icd10_code_nodot": icd10_code_nodot,
                        "icd10_family4": icd10_family4(ordo_row.get("reference", "")),
                        "icd9_code": "",
                        "icd9_code_nodot": "",
                        "icd9_family4": "",
                        "mapping_relation": ordo_row.get("mapping_relation", ""),
                        "mapping_icd_relation": ordo_row.get("mapping_icd_relation", ""),
                        "mapping_validation_status": ordo_row.get("mapping_validation_status", ""),
                        "gem_flags": "",
                        "gem_approximate": "",
                        "gem_no_map": "",
                        "gem_combination": "",
                        "gem_scenario": "",
                        "gem_choice_list": "",
                    })

    print(f"Wrote joined mapping to {output_path}")


if __name__ == "__main__":
    main()
