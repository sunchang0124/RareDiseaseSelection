#!/usr/bin/env python3.11
"""Guided CLI for the disease-selection pipeline.

Run with no arguments:
    python3.11 guided_disease_pipeline.py

The script walks through:
1. Disease lookup by ORDO code
2. Similar-disease search and selection
3. SQL generation
4. Optional cGAN data preparation if the SQL output CSV already exists
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pandas as pd

import prepare_cgan_data as pcd
from find_similar_diseases import find_similar

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "Data"
SQL_DIR = ROOT / "sql"
SELECTIONS_DIR = ROOT / "selections"
ORDO_DICT_PATH = ROOT.parent / "onto_cgans" / "data" / "ontology_emb" / "ORDO.dict"
ICD_MAP_PATH = ROOT / "OrdoICDMapping" / "ordo_icd10_icd9_mapping.csv"
COUNTS_PATH = ROOT / "OrdoICDMapping" / "ordo_patient_counts_subtype.csv"
CANDIDATE_CSV = ROOT / "candidate_diseases.csv"
SIMILAR_CSV = ROOT / "similar_diseases.csv"
ORDO_PREFIX = "http://www.orpha.net/ORDO/Orphanet_"
MIMIC_HOSP = "`physionet-data.mimiciv_3_1_hosp`"

SQL_DIR.mkdir(exist_ok=True)
SELECTIONS_DIR.mkdir(exist_ok=True)


def prompt(text: str, default: str | None = None) -> str:
    suffix = f" [{default}]" if default not in (None, "") else ""
    value = input(f"{text}{suffix}: ").strip()
    return value if value else (default or "")


def prompt_yes_no(text: str, default: bool = True) -> bool:
    default_label = "Y/n" if default else "y/N"
    value = input(f"{text} [{default_label}]: ").strip().lower()
    if not value:
        return default
    return value in {"y", "yes"}


def prompt_int(text: str, default: int, minimum: int | None = None, maximum: int | None = None) -> int:
    while True:
        raw = prompt(text, str(default))
        try:
            value = int(raw)
        except ValueError:
            print("Please enter an integer.")
            continue
        if minimum is not None and value < minimum:
            print(f"Please enter a value >= {minimum}.")
            continue
        if maximum is not None and value > maximum:
            print(f"Please enter a value <= {maximum}.")
            continue
        return value


def prompt_choice(text: str, choices: list[str], default: str) -> str:
    allowed = {choice.lower(): choice for choice in choices}
    while True:
        value = prompt(text, default).strip().lower()
        if value in allowed:
            return allowed[value]
        print(f"Please choose one of: {', '.join(choices)}.")


def load_ordo_dict() -> dict[str, str]:
    mapping: dict[str, str] = {}
    if ORDO_DICT_PATH.exists():
        with ORDO_DICT_PATH.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if ";" not in line:
                    continue
                name, uri = line.rsplit(";", 1)
                mapping[uri.strip()] = name.strip()
    return mapping


def load_icd_map() -> pd.DataFrame:
    if not ICD_MAP_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(ICD_MAP_PATH, dtype=str, low_memory=False).fillna("")


def load_mimic_counts() -> dict[str, int]:
    counts: dict[str, int] = {}
    if COUNTS_PATH.exists():
        df = pd.read_csv(COUNTS_PATH, dtype=str)
        counts = {row["ordo_code"]: int(row["n_unique_patients"]) for _, row in df.iterrows()}
    return counts


def load_candidate_csv() -> pd.DataFrame:
    if not CANDIDATE_CSV.exists():
        return pd.DataFrame(columns=["ordo_code", "disease_name", "n_unique_patients", "domain"])
    return pd.read_csv(CANDIDATE_CSV, dtype=str, quotechar='"', engine="python").fillna("")


def load_similar_csv() -> pd.DataFrame:
    if not SIMILAR_CSV.exists():
        return pd.DataFrame()
    return pd.read_csv(SIMILAR_CSV, dtype=str).fillna("")


def save_candidate(ordo_code: str, disease_name: str, n_patients: int, domain: str) -> None:
    df = load_candidate_csv()
    df = df[df["ordo_code"] != ordo_code]
    row = pd.DataFrame([{
        "ordo_code": ordo_code,
        "disease_name": disease_name,
        "n_unique_patients": str(n_patients),
        "domain": domain,
    }])
    pd.concat([df, row], ignore_index=True).to_csv(CANDIDATE_CSV, index=False, quoting=1)


def save_similar(target_ordo: str, target_name: str, target_n_label: str, rows: list[dict]) -> None:
    df = load_similar_csv()
    if not df.empty and "target_ordo" in df.columns:
        df = df[df["target_ordo"] != target_ordo]
    new_rows = [{
        "target_n": target_n_label,
        "target_ordo": target_ordo,
        "target_disease": target_name,
        "rank": str(i),
        "similar_ordo": row["ordo_code"],
        "similar_disease": row["disease_name"],
        "cosine_similarity": str(row["similarity"]),
        "n_mimic_patients": str(row["n_mimic_patients"]),
        "icd10_codes": row.get("icd10_codes", ""),
        "icd9_codes": row.get("icd9_codes", ""),
    } for i, row in enumerate(rows, start=1)]
    pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True).to_csv(SIMILAR_CSV, index=False)


def get_icd_codes(icd_map: pd.DataFrame, ordo_code: str) -> tuple[list[str], list[str]]:
    if icd_map.empty:
        return [], []
    rows = icd_map[icd_map["orpha_code"] == ordo_code]
    icd10 = sorted(set(rows["icd10_code_nodot"].str.strip()) - {""})
    icd9 = sorted(set(rows["icd9_code_nodot"].str.strip()) - {""})
    return icd10, icd9


def ordo_to_name(ordo_dict: dict[str, str], icd_map: pd.DataFrame, ordo_code: str) -> str:
    uri = f"{ORDO_PREFIX}{ordo_code}"
    name = ordo_dict.get(uri, "")
    if not name and not icd_map.empty:
        rows = icd_map[icd_map["orpha_code"] == ordo_code]
        if not rows.empty:
            name = rows["ordo_label"].iloc[0]
    return name


def filter_icd_codes(icd10: list[str], icd9: list[str], mode: str) -> tuple[list[str], list[str]]:
    if mode == "icd10":
        return icd10, []
    if mode == "icd9":
        return [], icd9
    return icd10, icd9


def generate_sql(target: dict, similar: list[dict], icd_mode: str = "both") -> str:
    def normalize_codes(codes: list[str]) -> list[str]:
        cleaned_codes = []
        for code in codes:
            cleaned = "".join(ch for ch in code.strip().upper() if ch.isalnum())
            if cleaned:
                cleaned_codes.append(cleaned)
        return cleaned_codes

    def minimize_prefixes(codes: list[str]) -> list[str]:
        unique_codes = sorted(set(normalize_codes(codes)), key=lambda code: (len(code), code))
        kept: list[str] = []
        for code in unique_codes:
            if any(code.startswith(existing) for existing in kept):
                continue
            kept.append(code)
        return kept

    def prefix_predicates(column: str, codes: list[str]) -> list[str]:
        kept = minimize_prefixes(codes)
        return [f"{column} LIKE '{code}%'" for code in kept]

    def remove_target_overlaps(similar_codes: list[str], target_codes: list[str]) -> list[str]:
        similar_min = minimize_prefixes(similar_codes)
        target_min = minimize_prefixes(target_codes)
        filtered = []
        for code in similar_min:
            overlaps_target = any(
                code.startswith(target_code) or target_code.startswith(code)
                for target_code in target_min
            )
            if not overlaps_target:
                filtered.append(code)
        return filtered

    def icd_where(icd10: list[str], icd9: list[str]) -> str:
        parts = []
        if icd10:
            code_filters = " OR ".join(prefix_predicates("d.icd_code", icd10))
            parts.append(f"(d.icd_version = 10 AND ({code_filters}))")
        if icd9:
            code_filters = " OR ".join(prefix_predicates("d.icd_code", icd9))
            parts.append(f"(d.icd_version = 9 AND ({code_filters}))")
        return " OR ".join(parts) if parts else "/* NO ICD CODES — verify mapping */ FALSE"

    union_parts = []
    target_icd10_min = minimize_prefixes(target["icd10_codes"])
    target_icd9_min = minimize_prefixes(target["icd9_codes"])

    for disease in [{"is_target": 1, **target}] + [{"is_target": 0, **row} for row in similar]:
        disease_name = disease["disease_name"].replace("'", "''")
        disease_icd10, disease_icd9 = filter_icd_codes(
            disease["icd10_codes"],
            disease["icd9_codes"],
            icd_mode,
        )
        if not disease["is_target"]:
            disease_icd10 = remove_target_overlaps(disease_icd10, target_icd10_min)
            disease_icd9 = remove_target_overlaps(disease_icd9, target_icd9_min)
        union_parts.append(f"""
    -- {'TARGET' if disease['is_target'] else 'SIMILAR'}: {disease['disease_name']}
    SELECT d.subject_id, d.hadm_id,
           '{disease_name}' AS disease_label,
           {disease['is_target']}  AS is_target,
           d.icd_code        AS matched_code
    FROM {MIMIC_HOSP}.diagnoses_icd d
    WHERE {icd_where(disease_icd10, disease_icd9)}""")

    return f"""-- onto-cGAN cohort query
-- Target  : {target['disease_name']} (ORDO {target['ordo_code']})
-- Similar : {', '.join(row['disease_name'] for row in similar)}
-- ICD mode: {icd_mode}
-- Generated: {date.today()}

WITH icd_hits AS ({"    UNION ALL".join(union_parts)}
),
cohort AS (
    SELECT DISTINCT subject_id, hadm_id, disease_label, is_target
    FROM icd_hits
),
matched_codes AS (
    SELECT subject_id, hadm_id,
           STRING_AGG(DISTINCT disease_label, ',' ORDER BY disease_label) AS all_matched_labels,
           STRING_AGG(DISTINCT matched_code,  ',' ORDER BY matched_code)  AS matched_icd_codes
    FROM icd_hits
    GROUP BY subject_id, hadm_id
),
demographics AS (
    SELECT a.subject_id, a.hadm_id, p.gender,
           a.admittime, a.dischtime,
           CAST(p.anchor_age + EXTRACT(YEAR FROM a.admittime) - p.anchor_year AS INT64) AS age
    FROM {MIMIC_HOSP}.admissions a
    JOIN {MIMIC_HOSP}.patients   p ON p.subject_id = a.subject_id
)
SELECT
    c.subject_id, c.hadm_id,
    c.disease_label, c.is_target,
    mc.all_matched_labels, mc.matched_icd_codes,
    dem.gender, dem.age, dem.admittime, dem.dischtime,
    di.label     AS lab_name,
    le.charttime AS lab_time,
    le.valuenum  AS lab_value
FROM cohort         c
JOIN matched_codes  mc  ON mc.subject_id  = c.subject_id  AND mc.hadm_id  = c.hadm_id
JOIN demographics   dem ON dem.subject_id = c.subject_id  AND dem.hadm_id = c.hadm_id
JOIN {MIMIC_HOSP}.labevents  le ON le.subject_id = c.subject_id AND le.hadm_id = c.hadm_id
JOIN {MIMIC_HOSP}.d_labitems di ON di.itemid = le.itemid
WHERE le.valuenum IS NOT NULL
ORDER BY c.subject_id, c.hadm_id, c.disease_label, le.charttime
;
"""


def parse_selection(raw: str, max_index: int) -> list[int]:
    raw = raw.strip().lower()
    if raw == "all":
        return list(range(1, max_index + 1))

    values: list[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue

        if "-" in part:
            start_raw, end_raw = part.split("-", 1)
            try:
                start = int(start_raw)
                end = int(end_raw)
            except ValueError:
                continue
            if start > end:
                start, end = end, start
            for idx in range(start, end + 1):
                if 1 <= idx <= max_index and idx not in values:
                    values.append(idx)
            continue

        try:
            idx = int(part)
        except ValueError:
            continue
        if 1 <= idx <= max_index and idx not in values:
            values.append(idx)
    return values


def save_selection_json(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
        fh.write("\n")


def load_selection_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def parse_ordo_code_list(raw: str) -> list[int]:
    raw = raw.strip()
    if not raw:
        return []
    if raw.startswith("[") and raw.endswith("]"):
        raw = raw[1:-1]
    values: list[int] = []
    for part in raw.split(","):
        part = part.strip().strip("'").strip('"')
        if not part:
            continue
        try:
            values.append(int(part))
        except ValueError:
            print(f"Skipping invalid ORDO code: {part}")
    deduped: list[int] = []
    seen: set[int] = set()
    for value in values:
        if value not in seen:
            deduped.append(value)
            seen.add(value)
    return deduped


def run_prepare_cgan(
    disease_csv: Path,
    out_stem: str,
    label_to_ordo: dict[str, int],
    strategies: list[str],
    exclude_ordo_codes: set[int],
    plot: bool,
    plot_by_disease: bool,
    tree_select: bool = True,
    top_features: int | None = None,
) -> list[Path]:
    original = dict(pcd.LABEL_TO_ORDO)
    pcd.LABEL_TO_ORDO.update(label_to_ordo)
    try:
        pcd.process_file(
            disease_csv,
            out_stem,
            DATA_DIR,
            strategies=strategies,
            plot=plot,
            max_features=30,
            plot_by_disease=plot_by_disease,
            exclude_ordo_codes=exclude_ordo_codes,
            tree_select=tree_select,
            top_features=top_features,
        )
    finally:
        pcd.LABEL_TO_ORDO.clear()
        pcd.LABEL_TO_ORDO.update(original)
    return [DATA_DIR / f"{out_stem}_{strategy}.csv" for strategy in strategies]


def prepare_from_selection(selection: dict) -> None:
    ordo_code = str(selection["ordo_code"])
    disease_name = selection["disease_name"]
    selected_similar = selection["selected_similar"]
    out_stem = selection.get("out_stem") or f"cgan_{ordo_code}"
    disease_csv = Path(selection.get("expected_input_csv", DATA_DIR / f"disease_{ordo_code}.csv"))

    print("\nPreparation step")
    print(f"Target disease : {disease_name}")
    print(f"ORDO code      : {ordo_code}")
    print(f"Input CSV      : {disease_csv}")
    print(f"Output stem    : {out_stem}")
    print(f"ICD mode       : {selection.get('icd_mode', 'both')}")

    if not disease_csv.exists():
        print("\nThe expected SQL result CSV does not exist yet.")
        print(f"Save the MIMIC query output here:\n  {disease_csv}")
        return

    if not prompt_yes_no("Run cGAN data preparation now", True):
        return

    label_to_ordo = {disease_name: int(ordo_code)}
    for row in selected_similar:
        label_to_ordo[row["disease_name"]] = int(row["ordo_code"])

    prep_mode = prompt_choice("Preparation mode: diagnosis, union, or both", ["diagnosis", "union", "both"], "diagnosis")
    strategies = ["diagnosis", "union"] if prep_mode == "both" else [prep_mode]
    include_all = prompt_yes_no("Include all diseases in the cohort", True)
    exclude_ordo_codes: set[int] = set()
    if include_all:
        print("All diseases will be kept.")
    else:
        raw_excluded = prompt("ORDO codes to exclude as a list like [123,456,789]", "[]")
        exclude_ordo_codes = set(parse_ordo_code_list(raw_excluded))
        print(f"Excluded ORDO codes: {sorted(exclude_ordo_codes)}")
    plot_by_disease = prompt_yes_no("Generate per-disease distribution plot too", False)
    tree_select = prompt_yes_no(
        "Run tree-based feature selection (Random Forest, target vs. similar)", True
    )
    top_features: int | None = None
    if tree_select:
        raw_top = prompt(
            "Number of top features to keep after ranking "
            "(leave blank to be prompted interactively after seeing the ranking)", ""
        ).strip()
        if raw_top:
            try:
                top_features = max(1, int(raw_top))
            except ValueError:
                print("Invalid number — you will be prompted interactively after the ranking.")
    out_paths = run_prepare_cgan(
        disease_csv,
        out_stem,
        label_to_ordo,
        strategies,
        exclude_ordo_codes,
        plot=True,
        plot_by_disease=plot_by_disease,
        tree_select=tree_select,
        top_features=top_features,
    )
    print("\nPrepared cGAN data:")
    for out_path in out_paths:
        print(f"  {out_path}")
    print("Plot paths:")
    for strategy in strategies:
        print(f"  {DATA_DIR / f'{out_stem}_{strategy}_feature_dist.png'}")
    if plot_by_disease:
        print("Per-disease plot paths:")
        for strategy in strategies:
            print(f"  {DATA_DIR / f'{out_stem}_{strategy}_feature_dist_by_disease.png'}")


def main() -> None:
    print("=" * 72)
    print("Guided disease-selection pipeline")
    print("You can just press Enter to accept the default at each step.")
    print("=" * 72)

    existing_selection = prompt("Existing ORDO selection to resume from (blank to start new)", "")
    if existing_selection:
        selection_path = SELECTIONS_DIR / f"{existing_selection}.json"
        if not selection_path.exists():
            raise SystemExit(f"Saved selection not found: {selection_path}")
        selection = load_selection_json(selection_path)
        action = prompt_choice("Jump to step: sql or prepare", ["sql", "prepare"], "prepare")
        if action == "sql":
            print(f"\nSaved SQL path:\n  {selection['sql_path']}")
            print(f"Expected MIMIC output CSV:\n  {selection['expected_input_csv']}")
            return
        prepare_from_selection(selection)
        return

    ordo_dict = load_ordo_dict()
    icd_map = load_icd_map()
    mimic_counts = load_mimic_counts()

    print("\nStep 1: Disease lookup")
    ordo_code = prompt("Enter ORDO code")
    icd_mode = prompt_choice("ICD source: icd9, icd10, or both", ["icd9", "icd10", "both"], "both")
    disease_name = ordo_to_name(ordo_dict, icd_map, ordo_code)
    if not disease_name:
        raise SystemExit(f"ORDO {ordo_code} not found in ORDO.dict or ICD mapping.")

    icd10_codes, icd9_codes = get_icd_codes(icd_map, ordo_code)
    shown_icd10, shown_icd9 = filter_icd_codes(icd10_codes, icd9_codes, icd_mode)
    n_mimic = mimic_counts.get(ordo_code, 0)

    print(f"\nTarget disease: {disease_name}")
    print(f"ORDO code     : {ordo_code}")
    print(f"MIMIC patients: {n_mimic}")
    print(f"ICD mode      : {icd_mode}")
    if shown_icd10:
        print(f"ICD-10 codes  : {', '.join(shown_icd10)}")
    if shown_icd9:
        print(f"ICD-9 codes   : {', '.join(shown_icd9)}")
    if not shown_icd10 and not shown_icd9:
        print("ICD codes     : none for the selected ICD mode")

    print("\nStep 2: Find similar diseases")
    use_saved = False
    saved_df = load_similar_csv()
    saved_rows = (
        saved_df[saved_df["target_ordo"] == ordo_code]
        if not saved_df.empty and "target_ordo" in saved_df.columns
        else pd.DataFrame()
    )
    if not saved_rows.empty:
        print(f"Found {len(saved_rows)} saved similar diseases for ORDO {ordo_code}.")
        use_saved = prompt_yes_no("Reuse saved similar diseases", True)

    if use_saved:
        similar_rows = [{
            "ordo_code": row["similar_ordo"],
            "disease_name": row["similar_disease"],
            "similarity": float(row["cosine_similarity"]),
            "n_mimic_patients": int(row["n_mimic_patients"] or 0),
            "icd10_codes": row["icd10_codes"],
            "icd9_codes": row["icd9_codes"],
        } for _, row in saved_rows.iterrows()]
        selected_similar = similar_rows
    else:
        topn = prompt_int("How many candidates should I fetch", 10, minimum=1, maximum=50)
        mimic_only = prompt_yes_no("Only include diseases with MIMIC patients", True)
        named_only = prompt_yes_no("Only include named diseases", True)
        similar_rows = find_similar(
            ordo_code=ordo_code,
            topn=topn,
            mimic_only=mimic_only,
            named_only=named_only,
        )
        for row in similar_rows:
            s_icd10, s_icd9 = get_icd_codes(icd_map, row["ordo_code"])
            row["icd10_codes"] = ", ".join(s_icd10)
            row["icd9_codes"] = ", ".join(s_icd9)

        if not similar_rows:
            raise SystemExit("No similar diseases found with the chosen filters.")

        print("\nCandidate similar diseases:")
        for i, row in enumerate(similar_rows, start=1):
            print(
                f"{i:>2}. ORDO {row['ordo_code']:<8}  "
                f"sim={row['similarity']:.3f}  "
                f"n={row['n_mimic_patients']:<6}  "
                f"{row['disease_name']}"
            )

        default_count = min(5, len(similar_rows))
        default_indices = f"1-{default_count}" if default_count > 1 else "1"
        while True:
            raw = prompt(
                "Select similar diseases by number or range",
                default_indices,
            )
            chosen = parse_selection(raw, len(similar_rows))
            if chosen:
                break
            print("Please choose at least one valid row number, range like 1-5, or 'all'.")

        selected_similar = [similar_rows[i - 1] for i in chosen]

    print("\nSelected similar diseases:")
    for row in selected_similar:
        print(f"- ORDO {row['ordo_code']}: {row['disease_name']} (n={row['n_mimic_patients']})")

    target_n_label = prompt("Target N label for saved metadata", f"~{n_mimic}")
    if prompt_yes_no("Save this similar-disease set to similar_diseases.csv", True):
        save_similar(ordo_code, disease_name, target_n_label, selected_similar)
        print(f"Saved -> {SIMILAR_CSV}")

    print("\nStep 3: Generate SQL")
    target = {
        "ordo_code": ordo_code,
        "disease_name": disease_name,
        "icd10_codes": shown_icd10,
        "icd9_codes": shown_icd9,
    }
    sql = generate_sql(target, [
        {
            "ordo_code": row["ordo_code"],
            "disease_name": row["disease_name"],
            "icd10_codes": [code.strip() for code in str(row.get("icd10_codes", "")).split(",") if code.strip()],
            "icd9_codes": [code.strip() for code in str(row.get("icd9_codes", "")).split(",") if code.strip()],
        }
        for row in selected_similar
    ], icd_mode=icd_mode)

    sql_path = SQL_DIR / f"{ordo_code}.sql"
    sql_path.write_text(sql, encoding="utf-8")
    print(f"SQL written to {sql_path}")

    out_stem = prompt("Output stem for cGAN file", f"cgan_{ordo_code}")
    disease_csv = DATA_DIR / f"disease_{ordo_code}.csv"
    selection_path = SELECTIONS_DIR / f"{ordo_code}.json"

    selection_payload = {
        "ordo_code": ordo_code,
        "disease_name": disease_name,
        "n_mimic": n_mimic,
        "icd_mode": icd_mode,
        "target_n_label": target_n_label,
        "icd10_codes": shown_icd10,
        "icd9_codes": shown_icd9,
        "sql_path": str(sql_path),
        "expected_input_csv": str(disease_csv),
        "out_stem": out_stem,
        "selected_similar": selected_similar,
    }
    save_selection_json(selection_path, selection_payload)
    print(f"Selection saved to {selection_path}")

    print("\nNext action")
    print(f"Run the SQL in MIMIC and save the result to:\n  {disease_csv}")

    if not disease_csv.exists():
        print("\nThat file does not exist yet, so data preparation is skipped for now.")
        print(f"When the CSV is ready, rerun this script and enter {ordo_code} at the resume prompt.")
        return

    prepare_from_selection(selection_payload)


if __name__ == "__main__":
    main()
