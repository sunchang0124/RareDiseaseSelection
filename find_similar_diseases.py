#!/usr/bin/env python3
"""Find the most similar diseases to a target ORDO disease using ontology embeddings.

The embeddings are Word2Vec vectors trained on the ORDO ontology graph.
Similarity is cosine similarity in embedding space.

Usage:
    python3.11 find_similar_diseases.py --ordo 29073
    python3.11 find_similar_diseases.py --ordo 29073 --topn 20
    python3.11 find_similar_diseases.py --ordo 29073 --mimic-only
    python3.11 find_similar_diseases.py --ordo 29073 --output similar_myeloma.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

EMBEDDINGS_PATH  = Path(__file__).parent.parent / "onto_cgans/data/ontology_emb/ontology.embeddings"
ORDO_DICT_PATH   = Path(__file__).parent.parent / "onto_cgans/data/ontology_emb/ORDO.dict"
ICD_MAPPING_PATH = Path(__file__).parent / "OrdoICDMapping" / "ordo_icd10_icd9_mapping.csv"
COUNTS_PATH      = Path(__file__).parent / "OrdoICDMapping" / "ordo_patient_counts_subtype.csv"
ORDO_URI_PREFIX  = "http://www.orpha.net/ORDO/Orphanet_"

_MODEL_CACHE = None
_URI_TO_NAME_CACHE: dict[str, str] | None = None
_MIMIC_COUNTS_CACHE: dict[str, int] | None = None


def load_ordo_dict(path: Path) -> dict[str, str]:
    """Return {uri: disease_name} from ORDO.dict (format: name;uri per line)."""
    mapping = {}
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or ";" not in line:
                continue
            name, uri = line.rsplit(";", 1)
            mapping[uri.strip()] = name.strip()
    return mapping


def load_icd_mapping_labels(path: Path) -> dict[str, str]:
    """Return {uri: disease_name} from ordo_icd10_icd9_mapping.csv.

    Used as a fallback when ORDO.dict does not contain an entry.
    """
    mapping = {}
    if not path.exists():
        return mapping
    with path.open("r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            code = row.get("orpha_code", "").strip()
            label = row.get("ordo_label", "").strip()
            if code and label:
                uri = f"{ORDO_URI_PREFIX}{code}"
                mapping.setdefault(uri, label)  # keep first occurrence
    return mapping


def load_mimic_counts(path: Path) -> dict[str, int]:
    """Return {orpha_code: n_unique_patients} from ordo_patient_counts_subtype.csv."""
    counts = {}
    if not path.exists():
        return counts
    with path.open("r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            counts[row["ordo_code"].strip()] = int(row["n_unique_patients"])
    return counts


def load_icd_counts(path: Path) -> dict[tuple[str, str], int]:
    """Return {(icd_version, icd_family4): n_unique_patients} from icd_patient_counts.csv."""
    counts: dict[tuple[str, str], int] = {}
    if not path.exists():
        return counts
    with path.open("r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            counts[(row["icd_version"].strip(), row["icd_family4"].strip())] = int(row["n_unique_patients"])
    return counts


def load_icd_family_map(path: Path) -> dict[str, dict[str, set[str]]]:
    """Return {ordo_code: {"10": {family4, ...}, "9": {family4, ...}}} from the ICD mapping CSV."""
    family_map: dict[str, dict[str, set[str]]] = {}
    if not path.exists():
        return family_map
    with path.open("r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            code = row.get("orpha_code", "").strip()
            if not code:
                continue
            entry = family_map.setdefault(code, {"10": set(), "9": set()})
            f10 = row.get("icd10_family4", "").strip()
            f9  = row.get("icd9_family4",  "").strip()
            if f10:
                entry["10"].add(f10)
            if f9:
                entry["9"].add(f9)
    return family_map


def count_ordo_patients(
    ordo_code: str,
    family_map: dict[str, dict[str, set[str]]],
    icd_counts: dict[tuple[str, str], int],
) -> int:
    """Sum ICD-family patient counts for an ORDO code."""
    entry = family_map.get(ordo_code, {})
    total = (sum(icd_counts.get(("10", f), 0) for f in entry.get("10", set())) +
             sum(icd_counts.get(("9",  f), 0) for f in entry.get("9",  set())))
    return total


def ordo_code_to_uri(code: str) -> str:
    return f"{ORDO_URI_PREFIX}{code.strip()}"


def uri_to_ordo_code(uri: str) -> str:
    return uri.replace(ORDO_URI_PREFIX, "").strip()


def load_embeddings():
    """Load the embedding model once per Python process."""
    global _MODEL_CACHE
    if _MODEL_CACHE is None:
        from gensim.models import Word2Vec
        _MODEL_CACHE = Word2Vec.load(str(EMBEDDINGS_PATH))
    return _MODEL_CACHE


def load_uri_to_name() -> dict[str, str]:
    """Load ORDO URI -> disease name once per process."""
    global _URI_TO_NAME_CACHE
    if _URI_TO_NAME_CACHE is None:
        uri_to_name = load_icd_mapping_labels(ICD_MAPPING_PATH)
        uri_to_name.update(load_ordo_dict(ORDO_DICT_PATH))
        _URI_TO_NAME_CACHE = uri_to_name
    return _URI_TO_NAME_CACHE


def load_mimic_counts_cached() -> dict[str, int]:
    """Load ORDO -> MIMIC patient counts once per process."""
    global _MIMIC_COUNTS_CACHE
    if _MIMIC_COUNTS_CACHE is None:
        _MIMIC_COUNTS_CACHE = load_mimic_counts(COUNTS_PATH)
    return _MIMIC_COUNTS_CACHE


def find_similar(
    ordo_code: str,
    topn: int = 10,
    mimic_only: bool = False,
    named_only: bool = False,
) -> list[dict]:
    """Return top-N similar ORDO diseases to the given ORDO code.

    Each result dict has keys:
        rank, ordo_code, disease_name, similarity, n_mimic_patients
    """
    target_uri = ordo_code_to_uri(ordo_code)

    model = load_embeddings()
    wv = model.wv

    if target_uri not in wv.key_to_index:
        raise KeyError(
            f"ORDO {ordo_code} (URI: {target_uri}) not found in embeddings vocabulary."
        )

    uri_to_name = load_uri_to_name()
    mimic_counts = load_mimic_counts_cached()

    # Retrieve a large pool of candidates, then filter to ORDO disease URIs only.
    # Ask for more than topn up front to have enough after filtering.
    pool_size = max(topn * 20, 500)
    raw_similar = wv.most_similar(target_uri, topn=pool_size)

    results = []
    for uri, sim in raw_similar:
        if not uri.startswith(ORDO_URI_PREFIX):
            continue
        code = uri_to_ordo_code(uri)
        n_patients = mimic_counts.get(code, 0)
        if mimic_only and n_patients == 0:
            continue
        name = uri_to_name.get(uri, "")
        if named_only and not name:
            continue
        results.append({
            "ordo_code": code,
            "disease_name": name or "unknown",
            "similarity": round(float(sim), 6),
            "n_mimic_patients": n_patients,
        })
        if len(results) >= topn:
            break

    # Add rank
    for i, r in enumerate(results, start=1):
        r["rank"] = i

    return results


def print_table(target_code: str, target_name: str, results: list[dict]) -> None:
    print(f"\nTop similar diseases to ORDO {target_code}: {target_name}\n")
    header = f"{'Rank':>4}  {'ORDO':>8}  {'Similarity':>10}  {'MIMIC N':>7}  Disease"
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r['rank']:>4}  {r['ordo_code']:>8}  {r['similarity']:>10.4f}"
            f"  {r['n_mimic_patients']:>7}  {r['disease_name']}"
        )


def write_csv(path: Path, target_code: str, target_name: str, results: list[dict]) -> None:
    fields = ["rank", "ordo_code", "disease_name", "similarity", "n_mimic_patients"]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)
    print(f"Wrote {len(results)} rows to {path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Find most similar ORDO diseases using ontology embeddings."
    )
    parser.add_argument(
        "--ordo",
        required=True,
        help="Target ORDO / Orphanet code (numeric, e.g. 29073)",
    )
    parser.add_argument(
        "--topn",
        type=int,
        default=10,
        help="Number of similar diseases to return (default: 10)",
    )
    parser.add_argument(
        "--mimic-only",
        action="store_true",
        help="Only return diseases that appear in MIMIC (n_mimic_patients > 0)",
    )
    parser.add_argument(
        "--named-only",
        action="store_true",
        help="Only return diseases with a resolvable name (skip anonymous ontology nodes)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional CSV output path",
    )
    parser.add_argument(
        "--embeddings",
        default=None,
        help="Override path to ontology.embeddings file",
    )
    parser.add_argument(
        "--ordo-dict",
        default=None,
        help="Override path to ORDO.dict file",
    )
    args = parser.parse_args()

    # Allow path overrides
    if args.embeddings:
        global EMBEDDINGS_PATH
        EMBEDDINGS_PATH = Path(args.embeddings)
    if args.ordo_dict:
        global ORDO_DICT_PATH
        ORDO_DICT_PATH = Path(args.ordo_dict)

    try:
        results = find_similar(
            ordo_code=args.ordo,
            topn=args.topn,
            mimic_only=args.mimic_only,
            named_only=args.named_only,
        )
    except KeyError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    # Resolve target disease name for display
    uri_to_name = load_icd_mapping_labels(ICD_MAPPING_PATH)
    uri_to_name.update(load_ordo_dict(ORDO_DICT_PATH))
    target_name = uri_to_name.get(ordo_code_to_uri(args.ordo), "unknown")

    print_table(args.ordo, target_name, results)

    if args.output:
        write_csv(Path(args.output), args.ordo, target_name, results)


if __name__ == "__main__":
    main()
