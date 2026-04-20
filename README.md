# RareDiseaseSelection

Pipeline to select unseen diseases for ontology-enhanced unseen disease generation using onto-cGAN.

Pipeline for building MIMIC-IV cohorts for rare diseases defined by ORDO ontology codes. The workflow maps MIMIC ICD codes to ORDO, finds ontologically similar diseases, generates BigQuery SQL for cohort extraction, and prepares wide-format cGAN-ready CSVs.

---

## Prerequisites

- Python 3.11
- Access to MIMIC-IV on BigQuery (`physionet-data.mimiciv_3_1_hosp`)
- ORDO ontology embeddings at `../onto_cgans/data/ontology_emb/` (see parent repo)

Install Python dependencies:

```bash
pip install pandas
```

---

## Required External Data Files

Download these two files and place them in `OrdoICDMapping/` before running the mapping build scripts.

### 1. Orphadata ICD mapping XML

```
OrdoICDMapping/en_product1.xml
```

Download from: https://www.orphadata.com/data/xml/en_product1.xml

### 2. ICD-10-CM to ICD-9-CM GEM crosswalk

```
OrdoICDMapping/icd10cmtoicd9gem.csv
```

Download from the CMS GEM files:
https://www.cms.gov/Medicare/Coding/ICD10/2018-ICD-10-CM-and-GEMs

---

## Setup: Build the ORDO–ICD Mapping

Run once after placing the external data files:

```bash
cd OrdoICDMapping
python3.11 extract_ordo_icd_mapping.py        # produces ordo_icd10_mapping.csv
python3.11 build_icd9_icd10_ordo_mapping.py   # produces ordo_icd10_icd9_mapping.csv
cd ..
cp OrdoICDMapping/ordo_icd10_icd9_mapping.csv ./ordo_icd10_icd9_mapping.csv
```

---

## Setup: Map MIMIC Diagnoses to ORDO

Place your MIMIC diagnosis extract at `Data/diag.csv`. Required columns:

| Column | Description |
|--------|-------------|
| `subject_id` | Patient identifier |
| `icd_code` | ICD code |
| `hadm_id` | Admission identifier |
| `icd_version` | `9` or `10` |

Then run:

```bash
python3.11 map_diag_to_ordo.py
```

This writes `OrdoICDMapping/diag_ordo_mapped.csv` and
`OrdoICDMapping/ordo_patient_counts_subtype.csv` (patient counts per ORDO code).

---

## Main Workflow: Guided Disease Pipeline

```bash
python3.11 guided_disease_pipeline.py
```

### Step-by-step flow

1. **Enter an ORDO code** (e.g. `519` for Acute myeloid leukemia)
2. **Choose ICD mode**: `icd9`, `icd10`, or `both`
3. **Review** the disease name, mapped ICD codes, and MIMIC patient count
4. **Select similar diseases** from ontology-neighbor suggestions
5. **SQL is written** to `sql/<ORDO>.sql` and session state saved to `selections/<ORDO>.json`
6. **Run the SQL** against MIMIC-IV on BigQuery and save the result to `Data/disease_<ORDO>.csv`
7. **Re-run the pipeline** for the same ORDO code — it detects the CSV and offers to run cGAN preprocessing

### Resuming a previous session

```bash
python3.11 guided_disease_pipeline.py
# enter the same ORDO code — the script loads selections/<ORDO>.json
# choose jump target: sql (regenerate SQL) or prepare (run preprocessing)
```

---

## cGAN Data Preparation

The `prepare_cgan_data.py` module is called automatically by the guided pipeline, but can also be run standalone:

```python
from prepare_cgan_data import process_file
process_file(
    in_path="Data/disease_519.csv",
    out_stem="cgan_519",
    out_dir="Data",
    strategies=["median"],   # or ["median", "first", "mean"]
)
```

### Output format

One wide-format CSV per strategy: `Data/<out_stem>_<strategy>.csv`

Columns: `IRI | label | gender | age | <lab columns>`

### Aggregation strategies

| Strategy | Description |
|----------|-------------|
| `median` (default) | Median of all lab draws within the first admission |
| `first` | Earliest lab draw per lab name within the first admission |
| `mean` | Mean of all lab draws within the first admission |

Preprocessing filters out lab columns with >30% missingness and patients with >30% missingness.

---

## File Structure

```
data_query/
├── guided_disease_pipeline.py   # main interactive CLI
├── prepare_cgan_data.py         # long → wide cGAN preprocessing
├── find_similar_diseases.py     # ontology-neighbor search
├── map_diag_to_ordo.py          # maps MIMIC diag.csv to ORDO
├── score_disease_candidates.py  # scoring helper for candidate diseases
├── pipeline_widgets.py          # Jupyter widget version (older)
│
├── OrdoICDMapping/
│   ├── extract_ordo_icd_mapping.py      # parses en_product1.xml
│   ├── build_icd9_icd10_ordo_mapping.py # builds final ICD-10/9 ↔ ORDO map
│   ├── en_product1.xml                  # [download — not in repo]
│   └── icd10cmtoicd9gem.csv             # [download — not in repo]
│
├── Data/          # place disease_<ORDO>.csv files here after BigQuery export
├── selections/    # auto-saved pipeline session states (JSON, gitignored)
└── sql/           # auto-generated MIMIC cohort queries (gitignored)
```

---

## ICD Matching Design

| Version | Match rule | Example |
|---------|-----------|---------|
| ICD-10 | letter + first 3 digits | `C9200 → C920` |
| ICD-9  | first 4 digits | `20500 → 2050` |

Family-level matching is used (not exact-code equality) to improve recall for codes with many sub-variants.

---

## Candidate Disease Targets

Selected targets for onto-cGAN experiments (lab-diagnosable, genuinely rare, ontologically connected):

| ORDO | Disease | Approx. MIMIC N | ICD-10 |
|------|---------|-----------------|--------|
| 519 | Acute myeloid leukemia | ~165 | C92.0 |
| 99819 | Familial gestational hyperthyroidism | ~570 | E05.8 |
| 54057 | Thrombotic thrombocytopenic purpura | — | M31.1 |
| 85443 | AL amyloidosis | ~114 | E85.4 |
| 29073 | Multiple myeloma | ~235 | C90.0 |
| 171 | Primary sclerosing cholangitis | ~320 | K83.0 |
