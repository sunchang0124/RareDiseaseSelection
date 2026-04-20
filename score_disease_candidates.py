#!/usr/bin/env python3.11
"""Score onto-cGAN disease candidates on three separate axes.

Axes:
1. ontology relatedness
2. ICD overlap risk
3. lab-distribution separability

The script prefers saved guided-pipeline selections when available so that the
neighbor set used for scoring matches the current working cohorts. It falls
back to `similar_diseases.csv` for targets without a saved selection.

Output:
- `disease_candidate_scores.csv` by default
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd

import prepare_cgan_data as pcd

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "Data"
MAPPING_PATH = ROOT / "OrdoICDMapping" / "ordo_icd10_icd9_mapping.csv"
COUNTS_PATH = ROOT / "OrdoICDMapping" / "ordo_patient_counts_subtype.csv"
DIAG_ORDO_PATH = ROOT / "OrdoICDMapping" / "diag_ordo_mapped.csv"
CANDIDATES_PATH = ROOT / "candidate_diseases.csv"
SIMILAR_PATH = ROOT / "similar_diseases.csv"
SELECTIONS_DIR = ROOT / "selections"
ORDO_PREFIX = "http://www.orpha.net/ORDO/Orphanet_"


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def safe_mean(values: list[float]) -> float | None:
    clean = [float(v) for v in values if pd.notna(v)]
    if not clean:
        return None
    return sum(clean) / len(clean)


def format_codes(values: set[str], limit: int = 8) -> str:
    if not values:
        return ""
    ordered = sorted(values)
    if len(ordered) <= limit:
        return ",".join(ordered)
    return ",".join(ordered[:limit]) + f",...(+{len(ordered) - limit})"


def load_candidates(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str).fillna("")
    if "ordo_code" not in df.columns:
        raise ValueError(f"Missing ordo_code column in {path}")
    return df


def augment_candidates(
    candidates: pd.DataFrame,
    similar_df: pd.DataFrame,
    label_map: dict[str, str],
    count_map: dict[str, int],
) -> pd.DataFrame:
    rows = []
    seen = {str(code).strip() for code in candidates["ordo_code"].astype(str)}

    for path in sorted(SELECTIONS_DIR.glob("*.json")):
        with path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
        code = str(payload.get("ordo_code", "")).strip()
        if not code or code in seen:
            continue
        seen.add(code)
        rows.append({
            "ordo_code": code,
            "disease_name": str(payload.get("disease_name", "")).strip() or label_map.get(code, ""),
            "n_unique_patients": str(payload.get("n_mimic", "")).strip() or str(count_map.get(code, "")),
            "domain": "",
        })

    if not similar_df.empty and "target_ordo" in similar_df.columns:
        for target_code in sorted(set(similar_df["target_ordo"].astype(str).str.strip()) - {""}):
            if target_code in seen:
                continue
            target_rows = similar_df[similar_df["target_ordo"].astype(str).str.strip() == target_code]
            target_name = ""
            if not target_rows.empty and "target_disease" in target_rows.columns:
                target_name = str(target_rows["target_disease"].iloc[0]).strip()
            seen.add(target_code)
            rows.append({
                "ordo_code": target_code,
                "disease_name": target_name or label_map.get(target_code, ""),
                "n_unique_patients": str(count_map.get(target_code, "")),
                "domain": "",
            })

    if not rows:
        return candidates
    return pd.concat([candidates, pd.DataFrame(rows)], ignore_index=True).fillna("")


def load_counts(path: Path) -> dict[str, int]:
    if not path.exists():
        return {}
    df = pd.read_csv(path, dtype=str).fillna("")
    return {
        str(row["ordo_code"]).strip(): int(row["n_unique_patients"])
        for _, row in df.iterrows()
        if str(row.get("ordo_code", "")).strip()
    }


def load_mapping(path: Path) -> tuple[dict[str, set[str]], dict[str, str]]:
    df = pd.read_csv(path, dtype=str, low_memory=False).fillna("")
    family_map: dict[str, set[str]] = {}
    label_map: dict[str, str] = {}
    for _, row in df.iterrows():
        code = row["orpha_code"].strip()
        if not code:
            continue
        label = row.get("ordo_label", "").strip()
        if label and code not in label_map:
            label_map[code] = label
        families = family_map.setdefault(code, set())
        icd10 = row.get("icd10_family4", "").strip()
        icd9 = row.get("icd9_family4", "").strip()
        if icd10:
            families.add(f"10:{icd10}")
        if icd9:
            families.add(f"9:{icd9}")
    return family_map, label_map


def load_subject_sets(path: Path) -> dict[str, set[str]]:
    if not path.exists():
        return {}
    df = pd.read_csv(path, dtype=str, usecols=["subject_id", "ordo_code"]).fillna("")
    grouped = df.groupby("ordo_code")["subject_id"]
    return {code: set(values) for code, values in grouped}


def load_saved_neighbors(target_code: str) -> tuple[list[dict], str | None]:
    selection_path = SELECTIONS_DIR / f"{target_code}.json"
    if not selection_path.exists():
        return [], None
    with selection_path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    rows = []
    for row in payload.get("selected_similar", []):
        rows.append({
            "ordo_code": str(row.get("ordo_code", "")).strip(),
            "disease_name": str(row.get("disease_name", "")).strip(),
            "similarity": float(row.get("similarity", 0.0) or 0.0),
            "n_mimic_patients": int(row.get("n_mimic_patients", 0) or 0),
        })
    return rows, payload.get("out_stem")


def load_fallback_neighbors(target_code: str, df: pd.DataFrame) -> list[dict]:
    if df.empty:
        return []
    rows = df[df["target_ordo"] == target_code].copy()
    if rows.empty:
        return []
    rows["rank_num"] = pd.to_numeric(rows["rank"], errors="coerce")
    rows = rows.sort_values(["rank_num", "similar_ordo"], kind="stable")
    out = []
    for _, row in rows.iterrows():
        out.append({
            "ordo_code": str(row["similar_ordo"]).strip(),
            "disease_name": str(row.get("similar_disease", "")).strip(),
            "similarity": float(row.get("cosine_similarity", 0.0) or 0.0),
            "n_mimic_patients": int(row.get("n_mimic_patients", 0) or 0),
        })
    return out


def dedupe_neighbors(rows: list[dict]) -> list[dict]:
    seen: set[str] = set()
    out: list[dict] = []
    for row in rows:
        code = row["ordo_code"]
        if not code or code in seen:
            continue
        seen.add(code)
        out.append(row)
    return out


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 0.0
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def overlap_fraction(reference: set[str], query: set[str]) -> float:
    if not query:
        return 0.0
    return len(reference & query) / len(query)


def load_cgan_or_prepare(target_code: str, out_stem_hint: str | None) -> tuple[pd.DataFrame | None, str]:
    candidates = []
    if out_stem_hint:
        candidates.append(DATA_DIR / f"{out_stem_hint}_union.csv")
    candidates.append(DATA_DIR / f"cgan_{target_code}_union.csv")
    for path in candidates:
        if path.exists():
            return pd.read_csv(path, low_memory=False), str(path)

    selection_path = SELECTIONS_DIR / f"{target_code}.json"
    if not selection_path.exists():
        return None, ""

    with selection_path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)

    disease_csv = Path(payload.get("expected_input_csv", ""))
    if not disease_csv.exists():
        return None, ""

    original = dict(pcd.LABEL_TO_ORDO)
    try:
        label_to_ordo = {payload["disease_name"]: int(payload["ordo_code"])}
        for row in payload.get("selected_similar", []):
            label_to_ordo[str(row["disease_name"])] = int(row["ordo_code"])
        pcd.LABEL_TO_ORDO.clear()
        pcd.LABEL_TO_ORDO.update(label_to_ordo)

        df = pcd._load_and_filter(disease_csv)
        wide = pcd._pivot(df.copy(), "union")
        wide = pcd._filter_missing(wide, threshold=0.30)
        wide = pcd._filter_missing_patients(wide, threshold=0.30)
        cgan, _ = pcd._to_cgan(wide)
        return cgan, f"{disease_csv} [prepared in-memory]"
    finally:
        pcd.LABEL_TO_ORDO.clear()
        pcd.LABEL_TO_ORDO.update(original)


def rankdata(values: pd.Series) -> pd.Series:
    return values.rank(method="average")


def fast_auc(y_true: pd.Series, scores: pd.Series) -> float | None:
    positives = int((y_true == 1).sum())
    negatives = int((y_true == 0).sum())
    if positives == 0 or negatives == 0:
        return None
    ranks = rankdata(scores.astype(float))
    sum_pos = float(ranks[y_true == 1].sum())
    auc = (sum_pos - positives * (positives + 1) / 2) / (positives * negatives)
    return auc


def best_single_feature_auc(df: pd.DataFrame, target_code: str) -> tuple[float | None, str, int]:
    if "IRI" not in df.columns:
        return None, "", 0
    work = df.copy()
    work["ordo_code"] = work["IRI"].astype(str).str.replace(ORDO_PREFIX, "", regex=False)
    work = work[work["ordo_code"] != ""].copy()
    if work.empty:
        return None, "", 0

    work["is_target_binary"] = (work["ordo_code"] == target_code).astype(int)
    if work["is_target_binary"].nunique() < 2:
        return None, "", int(work.shape[0])

    numeric_cols = []
    for col in work.columns:
        if col in {"IRI", "label", "gender", "age", "ordo_code", "is_target_binary"}:
            continue
        series = pd.to_numeric(work[col], errors="coerce")
        if series.notna().sum() >= 10:
            numeric_cols.append(col)
            work[col] = series

    best_auc: float | None = None
    best_feature = ""
    for col in numeric_cols:
        valid = work[["is_target_binary", col]].dropna()
        if valid["is_target_binary"].nunique() < 2 or len(valid) < 20:
            continue
        auc = fast_auc(valid["is_target_binary"], valid[col])
        if auc is None:
            continue
        auc = max(auc, 1.0 - auc)
        if best_auc is None or auc > best_auc:
            best_auc = auc
            best_feature = col

    return best_auc, best_feature, int(work.shape[0])


def separability_score_from_auc(auc: float | None) -> float | None:
    if auc is None:
        return None
    # Highest score for moderate separability around 0.75.
    score = 1.0 - abs(auc - 0.75) / 0.25
    return clamp01(score)


def ontology_score(rows: list[dict], topn: int, min_patients: int) -> tuple[float | None, float | None, int]:
    eligible = [
        row["similarity"]
        for row in rows
        if row["n_mimic_patients"] >= min_patients and row["similarity"] > 0
    ][:topn]
    if not eligible:
        return None, None, 0
    mean_sim = safe_mean(eligible)
    return mean_sim, mean_sim, len(eligible)


def icd_risk_score(
    target_code: str,
    neighbor_codes: list[str],
    family_map: dict[str, set[str]],
    subject_sets: dict[str, set[str]],
) -> dict[str, float | str | int | None]:
    target_families = family_map.get(target_code, set())
    target_subjects = subject_sets.get(target_code, set())

    family_jaccards: list[float] = []
    family_coverages: list[float] = []
    subject_jaccards: list[float] = []
    most_overlapping_codes: list[str] = []

    for code in neighbor_codes:
        neighbor_families = family_map.get(code, set())
        fj = jaccard(target_families, neighbor_families)
        fc = overlap_fraction(target_families, neighbor_families)
        family_jaccards.append(fj)
        family_coverages.append(fc)

        neighbor_subjects = subject_sets.get(code, set())
        if target_subjects or neighbor_subjects:
            subject_jaccards.append(jaccard(target_subjects, neighbor_subjects))

        if fj > 0:
            most_overlapping_codes.append(code)

    mean_family_jaccard = safe_mean(family_jaccards)
    mean_family_coverage = safe_mean(family_coverages)
    max_family_jaccard = max(family_jaccards) if family_jaccards else None
    mean_subject_jaccard = safe_mean(subject_jaccards)

    components = [v for v in [mean_family_jaccard, mean_family_coverage, mean_subject_jaccard] if v is not None]
    risk_score = safe_mean(components) if components else None

    return {
        "icd_overlap_risk_score": risk_score,
        "icd_family_jaccard_mean": mean_family_jaccard,
        "icd_family_overlap_fraction_mean": mean_family_coverage,
        "patient_overlap_jaccard_mean": mean_subject_jaccard,
        "max_family_jaccard": max_family_jaccard,
        "target_icd_families": format_codes(target_families),
        "neighbors_with_any_icd_overlap": ",".join(sorted(set(most_overlapping_codes))),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Score onto-cGAN disease candidates.")
    parser.add_argument("--candidates", default=str(CANDIDATES_PATH), help="Candidate disease CSV")
    parser.add_argument("--similar", default=str(SIMILAR_PATH), help="Saved similar diseases CSV")
    parser.add_argument("--mapping", default=str(MAPPING_PATH), help="ORDO ICD mapping CSV")
    parser.add_argument("--diag-ordo", default=str(DIAG_ORDO_PATH), help="Mapped diagnosis CSV")
    parser.add_argument("--counts", default=str(COUNTS_PATH), help="ORDO patient counts CSV")
    parser.add_argument("--output", default="disease_candidate_scores.csv", help="Output CSV path")
    parser.add_argument("--topn", type=int, default=5, help="Neighbors to summarize per target")
    parser.add_argument(
        "--min-neighbor-patients",
        type=int,
        default=1,
        help="Minimum MIMIC patients required for a neighbor to count",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    candidates = load_candidates(Path(args.candidates))
    similar_df = pd.read_csv(args.similar, dtype=str).fillna("") if Path(args.similar).exists() else pd.DataFrame()
    family_map, label_map = load_mapping(Path(args.mapping))
    subject_sets = load_subject_sets(Path(args.diag_ordo))
    count_map = load_counts(Path(args.counts))
    candidates = augment_candidates(candidates, similar_df, label_map, count_map)

    rows_out: list[dict] = []
    for _, row in candidates.iterrows():
        target_code = str(row["ordo_code"]).strip()
        target_name = str(row.get("disease_name", "")).strip() or label_map.get(target_code, "")
        target_n = int(count_map.get(target_code, 0))

        neighbors, out_stem = load_saved_neighbors(target_code)
        if not neighbors:
            neighbors = load_fallback_neighbors(target_code, similar_df)
            out_stem = None
        neighbors = dedupe_neighbors(neighbors)

        ontology_relatedness_score, ontology_similarity_mean, ontology_neighbor_count = ontology_score(
            neighbors,
            topn=args.topn,
            min_patients=args.min_neighbor_patients,
        )

        kept_neighbors = [
            n for n in neighbors
            if n["n_mimic_patients"] >= args.min_neighbor_patients
        ][:args.topn]
        neighbor_codes = [n["ordo_code"] for n in kept_neighbors]
        neighbor_names = [n["disease_name"] for n in kept_neighbors]

        icd_metrics = icd_risk_score(target_code, neighbor_codes, family_map, subject_sets)

        cgan_df, cgan_source = load_cgan_or_prepare(target_code, out_stem)
        auc, best_feature, cohort_rows = best_single_feature_auc(cgan_df, target_code) if cgan_df is not None else (None, "", 0)
        lab_sep_score = separability_score_from_auc(auc)

        rows_out.append({
            "target_ordo": target_code,
            "target_disease": target_name,
            "n_target_patients": target_n,
            "neighbors_used": len(neighbor_codes),
            "neighbor_ordos": ",".join(neighbor_codes),
            "neighbor_diseases": " | ".join(neighbor_names),
            "ontology_relatedness_score": ontology_relatedness_score,
            "ontology_similarity_mean": ontology_similarity_mean,
            "ontology_neighbors_counted": ontology_neighbor_count,
            **icd_metrics,
            "lab_distribution_separability_score": lab_sep_score,
            "lab_best_single_feature_auc": auc,
            "lab_best_feature": best_feature,
            "lab_cohort_rows": cohort_rows,
            "lab_score_source": cgan_source,
        })

    out_df = pd.DataFrame(rows_out)
    out_df = out_df.sort_values(
        by=["ontology_relatedness_score", "lab_distribution_separability_score"],
        ascending=[False, False],
        na_position="last",
    )
    out_df.to_csv(args.output, index=False)
    print(f"Wrote {len(out_df)} rows to {args.output}")


if __name__ == "__main__":
    main()
