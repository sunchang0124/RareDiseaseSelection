#!/usr/bin/env python3
"""Map MIMIC diagnosis rows to ORDO codes.

Input:
- Data/diag.csv
- OrdoICDMapping/ordo_icd10_icd9_mapping.csv

Output columns:
- subject_id
- icd_version
- icd_code
- ordo_code
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


OUTPUT_FIELDS = ["subject_id", "icd_version", "icd_code", "ordo_code", "parent_ordo_code"]
SUBTYPE_COUNT_FIELDS = ["ordo_code", "n_unique_patients"]
PARENT_COUNT_FIELDS = ["effective_ordo_code", "n_unique_patients"]


def normalize_code(code: str) -> str:
    return code.strip().upper().replace(".", "").replace(" ", "")


def icd9_family4(code: str) -> str:
    code = normalize_code(code)
    return code[:4]


def icd10_family4(code: str) -> str:
    code = normalize_code(code)
    return code[:4]


def normalize_label(label: str) -> str:
    return " ".join(label.strip().lower().split())


def build_parent_map(entries):
    """Infer a parent/general ORDO code within the same diagnosis family."""
    norm_labels = {code: normalize_label(label) for code, label in entries.items()}
    parents = {code: "" for code in entries}
    for code, label in norm_labels.items():
        if not label:
            continue
        candidates = []
        for other_code, other_label in norm_labels.items():
            if code == other_code or not other_label:
                continue
            if other_label != label and (label.startswith(other_label) or other_label in label):
                candidates.append((len(other_label), other_code))
        if candidates:
            candidates.sort()
            parents[code] = candidates[0][1]
    return parents


def load_mapping(path: Path):
    by_icd10 = {}
    by_icd9 = {}
    labels_icd10 = {}
    labels_icd9 = {}
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            ordo_code = (row.get("orpha_code") or "").strip()
            if not ordo_code:
                continue
            ordo_label = (row.get("ordo_label") or "").strip()

            icd10 = (row.get("icd10_family4") or "").strip()
            if icd10:
                by_icd10.setdefault(icd10, set()).add(ordo_code)
                labels_icd10.setdefault(icd10, {})[ordo_code] = ordo_label

            icd9 = (row.get("icd9_family4") or "").strip()
            if icd9:
                by_icd9.setdefault(icd9, set()).add(ordo_code)
                labels_icd9.setdefault(icd9, {})[ordo_code] = ordo_label

    parent_icd10 = {family: build_parent_map(entries) for family, entries in labels_icd10.items()}
    parent_icd9 = {family: build_parent_map(entries) for family, entries in labels_icd9.items()}
    return by_icd10, by_icd9, parent_icd10, parent_icd9


def main():
    parser = argparse.ArgumentParser(description="Map diag.csv to ORDO codes")
    parser.add_argument("--diag", default="Data/diag.csv", help="Path to diagnosis CSV")
    parser.add_argument(
        "--mapping",
        default="OrdoICDMapping/ordo_icd10_icd9_mapping.csv",
        help="Path to ORDO ICD10 ICD9 mapping CSV",
    )
    parser.add_argument(
        "--output",
        default="OrdoICDMapping/diag_ordo_mapped.csv",
        help="Output CSV path",
    )
    parser.add_argument(
        "--subtype-count-output",
        default="OrdoICDMapping/ordo_patient_counts_subtype.csv",
        help="Output CSV path for counts by raw ordo_code",
    )
    parser.add_argument(
        "--parent-count-output",
        default="OrdoICDMapping/ordo_patient_counts_parent.csv",
        help="Output CSV path for counts by parent-collapsed ORDO code",
    )
    parser.add_argument(
        "--counts-only",
        action="store_true",
        help="Only write count files; skip the full diag_ordo_mapped.csv output (faster)",
    )
    args = parser.parse_args()

    diag_path = Path(args.diag)
    mapping_path = Path(args.mapping)
    output_path = Path(args.output)
    subtype_count_output = Path(args.subtype_count_output)
    parent_count_output = Path(args.parent_count_output)

    by_icd10, by_icd9, parent_icd10, parent_icd9 = load_mapping(mapping_path)
    patients_by_ordo = {}
    patients_by_effective_ordo = {}

    if args.counts_only:
        with diag_path.open("r", encoding="utf-8", newline="") as diag_fh:
            for row in csv.DictReader(diag_fh):
                subject_id  = row.get("subject_id",  "").strip()
                icd_version = row.get("icd_version", "").strip()
                icd_code    = row.get("icd_code",    "").strip()

                if icd_version == "10":
                    family = icd10_family4(icd_code)
                    ordo_codes   = by_icd10.get(family, set())
                    parent_codes = parent_icd10.get(family, {})
                elif icd_version == "9":
                    family = icd9_family4(icd_code)
                    ordo_codes   = by_icd9.get(family, set())
                    parent_codes = parent_icd9.get(family, {})
                else:
                    continue

                for ordo_code in ordo_codes:
                    parent_ordo_code   = parent_codes.get(ordo_code, "")
                    effective_ordo_code = parent_ordo_code or ordo_code
                    patients_by_ordo.setdefault(ordo_code, set()).add(subject_id)
                    patients_by_effective_ordo.setdefault(effective_ordo_code, set()).add(subject_id)
    else:
        with diag_path.open("r", encoding="utf-8", newline="") as diag_fh, \
             output_path.open("w", encoding="utf-8", newline="") as out_fh:
            reader = csv.DictReader(diag_fh)
            writer = csv.DictWriter(out_fh, fieldnames=OUTPUT_FIELDS)
            writer.writeheader()

            for row in reader:
                subject_id  = row.get("subject_id",  "").strip()
                icd_version = row.get("icd_version", "").strip()
                icd_code    = row.get("icd_code",    "").strip()

                if icd_version == "10":
                    family = icd10_family4(icd_code)
                    ordo_codes   = by_icd10.get(family, set())
                    parent_codes = parent_icd10.get(family, {})
                elif icd_version == "9":
                    family = icd9_family4(icd_code)
                    ordo_codes   = by_icd9.get(family, set())
                    parent_codes = parent_icd9.get(family, {})
                else:
                    ordo_codes   = set()
                    parent_codes = {}

                for ordo_code in sorted(ordo_codes):
                    parent_ordo_code    = parent_codes.get(ordo_code, "")
                    effective_ordo_code = parent_ordo_code or ordo_code
                    patients_by_ordo.setdefault(ordo_code, set()).add(subject_id)
                    patients_by_effective_ordo.setdefault(effective_ordo_code, set()).add(subject_id)
                    writer.writerow({
                        "subject_id":       subject_id,
                        "icd_version":      icd_version,
                        "icd_code":         icd_code,
                        "ordo_code":        ordo_code,
                        "parent_ordo_code": parent_ordo_code,
                    })

        print(f"Wrote mapped diagnoses to {output_path}")

    with subtype_count_output.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=SUBTYPE_COUNT_FIELDS)
        writer.writeheader()
        sorted_subtype = sorted(
            patients_by_ordo.items(),
            key=lambda item: (-len(item[1]), item[0]),
        )
        for ordo_code, patient_set in sorted_subtype:
            writer.writerow({
                "ordo_code": ordo_code,
                "n_unique_patients": len(patient_set),
            })
    print(f"Wrote subtype counts to {subtype_count_output}")

    with parent_count_output.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=PARENT_COUNT_FIELDS)
        writer.writeheader()
        sorted_parent = sorted(
            patients_by_effective_ordo.items(),
            key=lambda item: (-len(item[1]), item[0]),
        )
        for effective_ordo_code, patient_set in sorted_parent:
            writer.writerow({
                "effective_ordo_code": effective_ordo_code,
                "n_unique_patients": len(patient_set),
            })
    print(f"Wrote parent-collapsed counts to {parent_count_output}")


if __name__ == "__main__":
    main()
