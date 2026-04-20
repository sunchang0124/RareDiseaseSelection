#!/usr/bin/env python3
"""Extract ORDO to ICD-10 mappings from an Orphadata XML file.

Reads the Orphadata XML product file and writes a flat CSV with one row per
Disorder x ICD-10 ExternalReference mapping entry.
"""

from __future__ import annotations

import argparse
import csv
import xml.etree.ElementTree as ET
from pathlib import Path


FIELDNAMES = [
    "orpha_code",
    "ordo_id",
    "disorder_name",
    "external_reference_id",
    "source",
    "reference",
    "reference_nodot",
    "mapping_relation",
    "mapping_icd_relation",
    "mapping_validation_status",
    "mapping_icd_ref_url",
    "mapping_icd_ref_uri",
]


def child_text(elem: ET.Element, tag: str) -> str:
    child = elem.find(tag)
    if child is None or child.text is None:
        return ""
    return child.text.strip()


def nested_name_text(elem: ET.Element, tag: str) -> str:
    parent = elem.find(tag)
    if parent is None:
        return ""
    name = parent.find("Name")
    if name is None or name.text is None:
        return ""
    return name.text.strip()


def normalize_code(code: str) -> str:
    return code.strip().upper().replace(".", "").replace(" ", "")


def iter_mapping_rows(xml_path: Path):
    for _, elem in ET.iterparse(xml_path, events=("end",)):
        if elem.tag != "Disorder":
            continue

        orpha_code = child_text(elem, "OrphaCode")
        disorder_name = ""
        name_elem = elem.find("Name")
        if name_elem is not None and name_elem.text is not None:
            disorder_name = name_elem.text.strip()
        ordo_id = f"http://www.orpha.net/ORDO/Orphanet_{orpha_code}" if orpha_code else ""

        ext_ref_list = elem.find("ExternalReferenceList")
        if ext_ref_list is not None:
            for ext_ref in ext_ref_list.findall("ExternalReference"):
                source = child_text(ext_ref, "Source")
                if source != "ICD-10":
                    continue
                yield {
                    "orpha_code": orpha_code,
                    "ordo_id": ordo_id,
                    "disorder_name": disorder_name,
                    "external_reference_id": ext_ref.attrib.get("id", ""),
                    "source": source,
                    "reference": child_text(ext_ref, "Reference"),
                    "reference_nodot": normalize_code(child_text(ext_ref, "Reference")),
                    "mapping_relation": nested_name_text(ext_ref, "DisorderMappingRelation"),
                    "mapping_icd_relation": nested_name_text(ext_ref, "DisorderMappingICDRelation"),
                    "mapping_validation_status": nested_name_text(ext_ref, "DisorderMappingValidationStatus"),
                    "mapping_icd_ref_url": child_text(ext_ref, "DisorderMappingICDRefUrl"),
                    "mapping_icd_ref_uri": child_text(ext_ref, "DisorderMappingICDRefUri"),
                }

        elem.clear()


def main():
    parser = argparse.ArgumentParser(description="Extract ORDO to ICD-10 mappings from Orphadata XML")
    parser.add_argument(
        "--input",
        default="en_product1.xml",
        help="Path to Orphadata XML file (default: en_product1.xml)",
    )
    parser.add_argument(
        "--output",
        default="ordo_icd10_mapping.csv",
        help="Output CSV path (default: ordo_icd10_mapping.csv)",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    with output_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in iter_mapping_rows(input_path):
            writer.writerow(row)

    print(f"Wrote mapping rows to {output_path}")


if __name__ == "__main__":
    main()
