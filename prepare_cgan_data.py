#!/usr/bin/env python3.11
"""
Prepare rare-disease cohort data for onto_cGAN training.

Produces one cGAN-ready CSV per disease (target + its similar diseases).
All queried lab features are kept — no coverage filtering.

Input  (long format — one row per patient × admission × lab measurement):
    Data/disease_1_fh.csv   — Homozygous FH + embedding-similar diseases
    Data/disease_2_psc.csv  — Primary sclerosing cholangitis + similar
    Data/disease_3_fgt.csv  — Familial gestational hyperthyroidism + similar
    Data/disease_4_mm.csv   — Multiple myeloma + similar
    Data/disease_5_al.csv   — AL amyloidosis + similar

Output (one file per disease):
    Data/cgan_fh.csv   — IRI | label | gender | age | <all FH labs>
    Data/cgan_psc.csv  — IRI | label | gender | age | <all PSC labs>
    Data/cgan_fgt.csv  — IRI | label | gender | age | <all FGT labs>
    Data/cgan_mm.csv   — IRI | label | gender | age | <all MM labs>
    Data/cgan_al.csv   — IRI | label | gender | age | <all AL labs>

Pipeline per file
-----------------
1. Load long-format CSV
2. Pivot: median over repeated lab draws per patient × admission
3. Best admission per (patient, disease_label): most non-null lab values
4. Map disease_label → ORDO code → IRI and label string
5. Output: [IRI | label | gender | age | all lab columns]

Usage
-----
    python3.11 prepare_cgan_data.py
    python3.11 prepare_cgan_data.py --out-dir Data/processed
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR    = Path(__file__).parent / "Data"
ORDO_PREFIX = "http://www.orpha.net/ORDO/Orphanet_"

# ── Disease files (input filename, output stem) ───────────────────────────────
DISEASE_FILES: list[tuple[str, str]] = [
    ("disease_1_fh.csv",  "cgan_fh"),
    ("disease_2_psc.csv", "cgan_psc"),
    ("disease_3_fgt.csv", "cgan_fgt"),
    ("disease_4_mm.csv",  "cgan_mm"),
    ("disease_5_al.csv",  "cgan_al"),
]

# ── disease_label (exact SQL CASE WHEN strings) → ORDO numeric code ──────────
LABEL_TO_ORDO: dict[str, int] = {
    # ── FH ───────────────────────────────────────────────────────────────
    "Homozygous FH":                                                       391665,
    "Familial cerebral saccular aneurysm":                                 231160,
    "Familial aortic dissection":                                          229,
    "ApoA-I deficiency / Abetalipoproteinemia":                            425,
    "Familial bicuspid aortic valve":                                      402075,
    "PPARG-related familial partial lipodystrophy":                        79083,
    "Familial exudative vitreoretinopathy":                                891,
    "Familial thoracic aortic aneurysm":                                   91387,
    # "Hyperlipidemia due to HTGL deficiency" (ORDO 140905) excluded — ICD E78.4/272.4
    # maps to 31k+ MIMIC patients, far too broad relative to FH target (~12k).
    # ── PSC ──────────────────────────────────────────────────────────────
    "Primary sclerosing cholangitis":                                      171,
    "Primary biliary cholangitis":                                         186,
    "PBC/PSC and autoimmune hepatitis overlap syndrome":                   562639,
    "Primary myelofibrosis":                                               824,
    "Pediatric systemic lupus erythematosus":                              93552,
    "Tibial muscular dystrophy":                                           609,
    "Oligoarticular juvenile idiopathic arthritis":                        85410,
    "Primary hepatic neuroendocrine carcinoma":                            100085,
    "Primary membranoproliferative glomerulonephritis":                    54370,
    "Intrahepatic cholestasis of pregnancy":                               69665,
    "Primary hypergonadotropic hypogonadism-partial alopecia syndrome":    2232,
    # ── FGT ──────────────────────────────────────────────────────────────
    "Familial gestational hyperthyroidism":                                99819,
    "Gestational choriocarcinoma":                                         99926,
    "Familial multinodular goiter":                                        276399,
    "Familial thyroid dyshormonogenesis / Hypothyroidism due to TSH receptor mutations": 95716,
    "Familial hypoaldosteronism":                                          427,
    "Familial chylomicronemia syndrome":                                   444490,
    "AKT2-related familial partial lipodystrophy":                         79085,
    "Thyroid lymphoma":                                                    97285,
    "X-linked non-progressive cerebellar ataxia":                         314978,
    # ── MM ───────────────────────────────────────────────────────────────
    "Multiple myeloma":                                                    29073,
    "Pyomyositis":                                                         764,
    "Adult-onset Still disease":                                           829,
    "Hypocalcemic vitamin D-resistant rickets":                            93160,
    "Mast cell sarcoma":                                                   66661,
    "Ollier disease":                                                      296,
    "Subcorneal pustular dermatosis":                                      48377,
    "Adamantinoma":                                                        55881,
    "Nephropathy-deafness-hyperparathyroidism syndrome":                   2668,
    "Multiple osteochondromas":                                            321,
    # ── AL ───────────────────────────────────────────────────────────────
    "AL amyloidosis":                                                      85443,
    "AApoAIV / Wild type ATTR amyloidosis":                                439232,
    "AA / Wild type ABeta2M amyloidosis":                                  85445,
    "Hereditary amyloidosis with primary renal involvement":               85450,
    "AGel / Variant ABeta2M amyloidosis":                                  85448,
    "Polyarteritis nodosa":                                                767,
    "Gaisbock syndrome":                                                   90041,
}

# ── Column names that are NOT lab features ────────────────────────────────────
META_COLS = {"subject_id", "hadm_id", "disease_label", "is_target",
             "ordo_code", "gender", "age"}


# ─────────────────────────────────────────────────────────────────────────────
def ordo_to_iri(code: int) -> str:
    return f"{ORDO_PREFIX}{code}"


def ordo_to_label(code: int) -> str:
    return f"ORDO.Orphanet_{code}"


# Aggregation strategies
# ─────────────────────────────────────────────────────────────────────────────
# Strategy A – union     : first non-null draw per lab across ALL matched admissions
# Strategy B – diagnosis : first non-null draw per lab within the earliest matched admission
# Strategy C – median    : median of all draws within the first admission only
# Strategy D – first     : earliest draw within the first admission only
# Strategy E – mean      : mean   of all draws within the first admission only
# ─────────────────────────────────────────────────────────────────────────────
STRATEGIES = {
    "union": "First non-null draw per lab across all matched admissions",
    "diagnosis": "First non-null draw per lab within the earliest matched admission",
    "median": "Median of all draws within the first admission",
    "first": "First (earliest) draw within the first admission",
    "mean": "Mean of all draws within the first admission",
}


def _load_and_filter(path: Path, exclude_ordo_codes: set[int] | None = None) -> pd.DataFrame:
    """Load CSV, drop unmapped labels, drop rows with no lab value.
    Does NOT filter by admission — strategies handle that in _pivot."""
    df = pd.read_csv(path, low_memory=False)
    print(f"  Input : {len(df):,} rows | "
          f"{df['subject_id'].nunique():,} patients | "
          f"{df['hadm_id'].nunique():,} admissions")

    unknown = df.loc[~df["disease_label"].isin(LABEL_TO_ORDO), "disease_label"].unique()
    if len(unknown):
        print(f"  WARNING: unmapped labels excluded: {unknown.tolist()}")
        df = df[df["disease_label"].isin(LABEL_TO_ORDO)].copy()

    if exclude_ordo_codes:
        excluded_labels = sorted(
            label for label, code in LABEL_TO_ORDO.items()
            if int(code) in exclude_ordo_codes
        )
        if excluded_labels:
            before = len(df)
            df = df[~df["disease_label"].isin(excluded_labels)].copy()
            print(
                "  Excluded ORDO codes : "
                f"{sorted(exclude_ordo_codes)} -> removed {before - len(df):,} rows"
            )

    df = df.dropna(subset=["lab_name", "lab_value"])
    df["admittime"] = pd.to_datetime(df["admittime"])
    df["lab_time"]  = pd.to_datetime(df["lab_time"])
    return df


def _pivot(df: pd.DataFrame, strategy: str) -> pd.DataFrame:
    """Pivot long → wide using the chosen aggregation strategy."""

    if strategy == "union":
        # First non-null draw per (patient, disease_label, lab) across ALL admissions.
        # Sort chronologically so the earliest measurement is kept.
        df_sorted = df.sort_values(["admittime", "lab_time"])
        df_dedup = df_sorted.drop_duplicates(
            subset=["subject_id", "disease_label", "lab_name"], keep="first"
        )
        wide = df_dedup.pivot_table(
            index=["subject_id", "disease_label", "is_target", "gender", "age"],
            columns="lab_name",
            values="lab_value",
            aggfunc="first",
        ).reset_index()
        wide.columns.name = None
        return wide

    if strategy == "diagnosis":
        # Keep labs only from the earliest diagnosis-matched admission for each
        # (patient, disease_label), then take the earliest non-null draw per lab.
        first_admit = (
            df.groupby(["subject_id", "disease_label"])["admittime"]
            .min().reset_index()
            .rename(columns={"admittime": "first_admittime"})
        )
        df = df.merge(first_admit, on=["subject_id", "disease_label"])
        df = df[df["admittime"] == df["first_admittime"]].drop(columns="first_admittime")
        df = (
            df.sort_values(["subject_id", "disease_label", "admittime", "lab_time"])
            .drop_duplicates(
                subset=["subject_id", "disease_label", "hadm_id", "lab_name"],
                keep="first",
            )
        )
        wide = df.pivot_table(
            index=["subject_id", "hadm_id", "disease_label", "is_target", "gender", "age"],
            columns="lab_name",
            values="lab_value",
            aggfunc="first",
        ).reset_index()
        wide.columns.name = None
        return wide.drop(columns="hadm_id")

    # ── First-admission strategies ────────────────────────────────────────────
    first_admit = (
        df.groupby(["subject_id", "disease_label"])["admittime"]
        .min().reset_index()
        .rename(columns={"admittime": "first_admittime"})
    )
    df = df.merge(first_admit, on=["subject_id", "disease_label"])
    df = df[df["admittime"] == df["first_admittime"]].drop(columns="first_admittime")

    index_cols = ["subject_id", "hadm_id", "disease_label", "is_target", "gender", "age"]
    if strategy == "first":
        df = (df.sort_values("lab_time")
                .drop_duplicates(subset=index_cols + ["lab_name"], keep="first"))
        aggfunc = "first"
    elif strategy == "mean":
        aggfunc = "mean"
    else:  # median
        aggfunc = "median"

    wide = df.pivot_table(
        index=index_cols,
        columns="lab_name",
        values="lab_value",
        aggfunc=aggfunc,
    ).reset_index()
    wide.columns.name = None
    return wide.drop(columns="hadm_id")


def _print_missing(cgan: pd.DataFrame, lab_cols: list[str], top_n: int = 20) -> None:
    """Print overall missing rate and the top_n most-missing lab features."""
    missing = (cgan[lab_cols].isna().mean() * 100).sort_values(ascending=False)
    overall = missing.mean()
    print(f"\n  Overall missing rate : {overall:.1f}%  "
          f"({int(cgan[lab_cols].isna().sum().sum()):,} / "
          f"{len(cgan) * len(lab_cols):,} cells)")
    print(f"  Top {top_n} most-missing features:")
    print(f"  {'Lab feature':50s}  {'Missing':>8}")
    print(f"  {'─'*61}")
    for lab, pct in missing.head(top_n).items():
        print(f"  {str(lab)[:50]:50s}  {pct:>7.1f}%")


def _filter_missing(wide: pd.DataFrame, threshold: float = 0.30) -> pd.DataFrame:
    """Drop lab columns where the fraction of missing values exceeds threshold.

    Only non-metadata columns are evaluated; metadata columns are always kept.
    Prints a one-line summary of how many features were removed.
    """
    lab_cols = [c for c in wide.columns if c not in META_COLS]
    missing_frac = wide[lab_cols].isna().mean()
    keep = missing_frac[missing_frac <= threshold].index.tolist()
    drop = missing_frac[missing_frac >  threshold].index.tolist()
    print(f"\n  Missing filter (>{threshold*100:.0f}%): "
          f"dropped {len(drop)} features, kept {len(keep)} features")
    return wide.drop(columns=drop)


def _filter_missing_patients(wide: pd.DataFrame, threshold: float = 0.30) -> pd.DataFrame:
    """Drop rows where the fraction of missing lab values exceeds threshold.

    Only lab (non-metadata) columns count toward the missing fraction.
    Prints a one-line summary of how many patients were removed.
    """
    lab_cols = [c for c in wide.columns if c not in META_COLS]
    missing_frac = wide[lab_cols].isna().mean(axis=1)
    keep_mask = missing_frac <= threshold
    target_mask = wide["is_target"] == 1

    # Preserve target rows when the filter would otherwise eliminate the
    # target cohort entirely; downstream plotting and model prep need at
    # least some target examples to remain.
    if target_mask.any() and not (keep_mask & target_mask).any():
        rescued = int((~keep_mask & target_mask).sum())
        keep_mask = keep_mask | target_mask
        print(f"  Target rescue    : kept {rescued} target row(s) that exceeded missingness threshold")

    n_dropped = (~keep_mask).sum()
    print(f"  Patient filter   (>{threshold*100:.0f}%): "
          f"dropped {n_dropped} patients, kept {keep_mask.sum()}")
    return wide[keep_mask].reset_index(drop=True)


def plot_feature_distributions(
    wide: pd.DataFrame,
    out_stem: str,
    out_dir: Path,
    strategy: str,
    max_features: int = 30,
) -> None:
    """Plot KDE distributions of lab features: target disease vs. similar diseases pooled.

    One PNG is saved per (disease file, strategy).  Features are sorted by
    overall coverage (fewest missing values first) and the top `max_features`
    are shown in a grid of sub-plots.

    Parameters
    ----------
    wide:
        Wide-format dataframe that still contains the ``is_target`` and
        ``disease_label`` meta-columns (i.e. before ``_to_cgan`` strips them).
    out_stem:
        Output filename stem (e.g. ``"cgan_fh"``).
    out_dir:
        Directory where the PNG is written.
    strategy:
        Aggregation strategy name used as part of the filename / title.
    max_features:
        Maximum number of features to display (default 30).
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  WARNING: matplotlib not available — skipping feature distribution plot.")
        return

    lab_cols = [c for c in wide.columns if c not in META_COLS]

    target_mask  = wide["is_target"] == 1
    similar_mask = wide["is_target"] == 0

    if not target_mask.any():
        print("  WARNING: no target-disease rows found — skipping plot.")
        return

    target_name = wide.loc[target_mask, "disease_label"].iloc[0]
    n_target    = int(target_mask.sum())
    n_similar   = int(similar_mask.sum())

    # Rank features by overall non-null coverage, take the top N
    coverage  = wide[lab_cols].notna().mean().sort_values(ascending=False)
    plot_cols = coverage.head(max_features).index.tolist()

    ncols = 5
    nrows = max(1, (len(plot_cols) + ncols - 1) // ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.5, nrows * 2.8))
    axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for i, col in enumerate(plot_cols):
        ax = axes_flat[i]

        target_vals  = wide.loc[target_mask,  col].dropna()
        similar_vals = wide.loc[similar_mask, col].dropna()

        plotted = False
        if len(target_vals) >= 5:
            target_vals.plot.kde(
                ax=ax,
                label=f"Target (n={len(target_vals)})",
                color="crimson",
                linewidth=1.8,
            )
            plotted = True
        if len(similar_vals) >= 5:
            similar_vals.plot.kde(
                ax=ax,
                label=f"Similar (n={len(similar_vals)})",
                color="steelblue",
                linewidth=1.8,
                alpha=0.75,
            )
            plotted = True

        if not plotted:
            ax.text(0.5, 0.5, "insufficient data",
                    ha="center", va="center", transform=ax.transAxes, fontsize=7)

        ax.set_title(str(col)[:38], fontsize=7, pad=3)
        ax.set_xlabel("")
        ax.set_ylabel("Density", fontsize=6)
        ax.tick_params(labelsize=6)
        ax.legend(fontsize=5, loc="upper right")

    # Hide unused subplot axes
    for j in range(len(plot_cols), len(axes_flat)):
        axes_flat[j].set_visible(False)

    similar_labels = sorted(wide.loc[similar_mask, "disease_label"].unique())
    similar_str    = ", ".join(similar_labels)

    fig.suptitle(
        f"Target: {target_name}  (n={n_target})\n"
        f"vs. similar diseases pooled  (n={n_similar})  —  {strategy.upper()} strategy\n"
        f"Similar: {similar_str[:120]}{'…' if len(similar_str) > 120 else ''}",
        fontsize=8,
        y=1.01,
    )
    fig.tight_layout()

    out_path = out_dir / f"{out_stem}_{strategy}_feature_dist.png"
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  Feature distribution plot → {out_path.name}")


def plot_feature_distributions_by_disease(
    wide: pd.DataFrame,
    out_stem: str,
    out_dir: Path,
    strategy: str,
    max_features: int = 20,
    max_diseases: int = 6,
) -> None:
    """Plot KDE distributions for each disease separately on shared axes.

    Saves one PNG per (disease file, strategy), overlaying up to `max_diseases`
    disease labels for the top `max_features` highest-coverage lab features.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  WARNING: matplotlib not available — skipping per-disease distribution plot.")
        return

    lab_cols = [c for c in wide.columns if c not in META_COLS]
    disease_counts = wide["disease_label"].value_counts()
    disease_labels = disease_counts.head(max_diseases).index.tolist()
    if len(disease_labels) < 2:
        print("  WARNING: need at least 2 diseases for per-disease plot — skipping.")
        return

    coverage = wide[lab_cols].notna().mean().sort_values(ascending=False)
    plot_cols = coverage.head(max_features).index.tolist()

    ncols = 4
    nrows = max(1, (len(plot_cols) + ncols - 1) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4.0, nrows * 3.0))
    axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]
    cmap = plt.get_cmap("tab10")

    for i, col in enumerate(plot_cols):
        ax = axes_flat[i]
        plotted_any = False
        for j, disease_label in enumerate(disease_labels):
            vals = wide.loc[wide["disease_label"] == disease_label, col].dropna()
            if len(vals) < 5:
                continue
            vals.plot.kde(
                ax=ax,
                label=f"{disease_label[:28]} (n={len(vals)})",
                color=cmap(j % 10),
                linewidth=1.6,
                alpha=0.85,
            )
            plotted_any = True

        if not plotted_any:
            ax.text(0.5, 0.5, "insufficient data",
                    ha="center", va="center", transform=ax.transAxes, fontsize=7)

        ax.set_title(str(col)[:38], fontsize=7, pad=3)
        ax.set_xlabel("")
        ax.set_ylabel("Density", fontsize=6)
        ax.tick_params(labelsize=6)
        ax.legend(fontsize=5, loc="upper right")

    for j in range(len(plot_cols), len(axes_flat)):
        axes_flat[j].set_visible(False)

    disease_summary = ", ".join(f"{label} (n={int(disease_counts[label])})" for label in disease_labels)
    fig.suptitle(
        f"Per-disease feature distributions — {strategy.upper()} strategy\n"
        f"Diseases shown: {disease_summary[:180]}{'…' if len(disease_summary) > 180 else ''}",
        fontsize=8,
        y=1.01,
    )
    fig.tight_layout()

    out_path = out_dir / f"{out_stem}_{strategy}_feature_dist_by_disease.png"
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  Per-disease feature distribution plot → {out_path.name}")


def _tree_select_features(
    wide: pd.DataFrame,
    top_n: int | None,
) -> pd.DataFrame:
    """Rank lab features by Random Forest importance (target vs. similar diseases).

    Trains a binary RF classifier on is_target, prints the ranked importance
    table, then keeps only the top_n features.  If top_n is None the user is
    prompted interactively (defaults to 20 when stdin is not a TTY).

    Parameters
    ----------
    wide:
        Wide dataframe still containing is_target and all meta columns.
    top_n:
        Number of top features to keep.  None → ask interactively.
    """
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.impute import SimpleImputer
    except ImportError:
        print("  WARNING: scikit-learn not available — skipping tree-based feature selection.")
        return wide

    lab_cols = [c for c in wide.columns if c not in META_COLS]

    if len(lab_cols) == 0:
        print("  WARNING: no lab columns found — skipping tree-based feature selection.")
        return wide

    y = wide["is_target"].values
    X = wide[lab_cols].values

    # Impute missing values with column median before fitting
    imputer = SimpleImputer(strategy="median")
    X_imp = imputer.fit_transform(X)

    clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    clf.fit(X_imp, y)

    importances = pd.Series(clf.feature_importances_, index=lab_cols)
    ranked = importances.sort_values(ascending=False)

    print(f"\n  Tree-based feature importance (target vs. similar, {len(lab_cols)} features):")
    print(f"  {'Rank':>4}  {'Lab feature':50s}  {'Importance':>10}")
    print(f"  {'─'*68}")
    for rank, (feat, imp) in enumerate(ranked.items(), start=1):
        marker = " <--" if rank == 20 else ""
        print(f"  {rank:>4}  {str(feat)[:50]:50s}  {imp:>10.4f}{marker}")
    print(f"  {'─'*68}")

    if top_n is None:
        if sys.stdin.isatty():
            try:
                answer = input("\n  How many top features to keep? [default: 20] ").strip()
                top_n = int(answer) if answer else 20
            except (ValueError, EOFError):
                print("  Invalid input — using default of 20.")
                top_n = 20
        else:
            print("  Non-interactive mode — using default of 20 top features.")
            top_n = 20

    top_n = max(1, min(top_n, len(lab_cols)))
    keep_labs = ranked.head(top_n).index.tolist()
    dropped = len(lab_cols) - top_n
    print(f"\n  Tree selection: keeping top {top_n} features, dropping {dropped}")

    keep_cols = [c for c in wide.columns if c in META_COLS or c in keep_labs]
    return wide[keep_cols]


def _to_cgan(wide: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Encode gender, add IRI/label columns, return (cgan_df, lab_cols)."""
    wide["gender"] = (
        wide["gender"].map({"M": "Male", "F": "Female"})
        .fillna(wide["gender"])
    )
    wide["ordo_code"] = wide["disease_label"].map(LABEL_TO_ORDO)
    iri_series   = wide["ordo_code"].apply(lambda x: ordo_to_iri(int(x)))
    label_series = wide["ordo_code"].apply(lambda x: ordo_to_label(int(x)))

    lab_cols = [c for c in wide.columns if c not in META_COLS]
    cgan = wide[["gender", "age"] + lab_cols].copy()
    cgan.insert(0, "IRI",   iri_series)
    cgan.insert(1, "label", label_series)
    return cgan, lab_cols


def process_file(path: Path, out_stem: str, out_dir: Path,
                 strategies: list[str], plot: bool = False,
                 max_features: int = 30,
                 plot_by_disease: bool = False,
                 exclude_ordo_codes: set[int] | None = None,
                 tree_select: bool = False,
                 top_features: int | None = None) -> None:
    """
    Load one long-format disease CSV and save one cGAN-ready CSV per strategy.

    Output filenames: <out_stem>_<strategy>.csv
    If ``plot`` is True, also saves a target-vs-pooled feature-distribution PNG.
    If ``plot_by_disease`` is True, also saves a per-disease distribution PNG.
    If ``tree_select`` is True, runs Random Forest importance ranking after the
    missingness filter and keeps only the top features.
    """
    print(f"\n{'─'*60}")
    print(f"  {path.name}")
    print(f"{'─'*60}")

    df = _load_and_filter(path, exclude_ordo_codes=exclude_ordo_codes)

    for strategy in strategies:
        wide = _pivot(df.copy(), strategy)
        wide = _filter_missing(wide, threshold=0.30)
        wide = _filter_missing_patients(wide, threshold=0.30)

        if tree_select:
            wide = _tree_select_features(wide, top_n=top_features)

        if plot:
            plot_feature_distributions(wide, out_stem, out_dir, strategy,
                                       max_features=max_features)
        if plot_by_disease:
            plot_feature_distributions_by_disease(wide, out_stem, out_dir, strategy,
                                                  max_features=min(max_features, 20))

        cgan, lab_cols = _to_cgan(wide)

        out_path = out_dir / f"{out_stem}_{strategy}.csv"
        cgan.to_csv(out_path, index=False)

        # Summary
        print(f"\n  [{strategy.upper()}] {STRATEGIES[strategy]}")
        _print_missing(cgan, lab_cols)
        print(f"  {'Label':55s}  {'ORDO':>8}  {'N':>6}")
        print(f"  {'─'*72}")
        for lbl, grp in sorted(wide.groupby("disease_label"), key=lambda x: x[0]):
            ordo = int(grp["ordo_code"].iloc[0])
            print(f"  {lbl[:55]:55s}  {ordo:>8}  {len(grp):>6}")
        print(f"  {'─'*72}")
        print(f"  {'TOTAL':55s}  {'':8}  {len(wide):>6}")
        print(f"  → {out_path.name}  ({cgan.shape[0]:,} rows × {cgan.shape[1]} cols, "
              f"{len(lab_cols)} lab features)")


# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare per-disease cohort data for onto_cGAN."
    )
    parser.add_argument(
        "--out-dir", default=None,
        help="Output directory (default: same as Data/)",
    )
    parser.add_argument(
        "--strategies", nargs="+", default=["union"],
        choices=list(STRATEGIES.keys()),
        help="Within-admission aggregation strategies to produce (default: union).",
    )
    parser.add_argument(
        "--plot", action="store_true",
        help="Save feature-distribution KDE plots (target vs. similar diseases).",
    )
    parser.add_argument(
        "--exclude-ordo-codes", nargs="*", type=int, default=None,
        help="Optional ORDO codes to exclude before preprocessing.",
    )
    parser.add_argument(
        "--max-features", type=int, default=30,
        help="Maximum number of features shown in each distribution plot (default: 30).",
    )
    parser.add_argument(
        "--no-tree-select", dest="tree_select", action="store_false",
        help="Disable tree-based feature selection (enabled by default).",
    )
    parser.set_defaults(tree_select=True)
    parser.add_argument(
        "--top-features", type=int, default=None,
        help=(
            "Number of top-ranked features to keep when --tree-select is enabled. "
            "If omitted you will be prompted interactively (default when non-interactive: 20)."
        ),
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else DATA_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Preparing per-disease cGAN input files")
    print(f"Strategies: {args.strategies}")
    if args.plot:
        print(f"Plots: enabled (max {args.max_features} features per plot)")
    if args.tree_select:
        top_str = str(args.top_features) if args.top_features else "interactive"
        print(f"Tree-based feature selection: enabled (top features: {top_str})")
    print("=" * 60)

    for in_file, out_stem in DISEASE_FILES:
        in_path = DATA_DIR / in_file
        process_file(in_path, out_stem, out_dir, args.strategies,
                     plot=args.plot, max_features=args.max_features,
                     exclude_ordo_codes=set(args.exclude_ordo_codes or []),
                     tree_select=args.tree_select,
                     top_features=args.top_features)

    print(f"\n{'='*60}")
    print("Done. Output files:")
    for _, out_stem in DISEASE_FILES:
        for strategy in args.strategies:
            p = out_dir / f"{out_stem}_{strategy}.csv"
            if p.exists():
                n = sum(1 for _ in p.open()) - 1
                print(f"  {p.name:30s}  {n:,} rows")
    print("=" * 60)


if __name__ == "__main__":
    main()
