"""Query ChEMBL for hERG/KCNH2 assays and export raw + cleaned activity tables.

Usage:
    python query.py
    python query.py --raw-out data/herg_raw.tsv --clean-out data/herg_clean.tsv --sep "\t"
"""

from __future__ import annotations

import argparse
import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple

import pandas as pd

try:
    from chembl_webresource_client.new_client import new_client
except Exception as exc:  # pragma: no cover - import error is runtime/environmental
    raise RuntimeError(
        "chembl_webresource_client is required. Install with "
        "`pip install chembl_webresource_client`."
    ) from exc


DEFAULT_TERMS: Tuple[str, ...] = ("hERG", "HERG", "KCNH2")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract hERG/KCNH2 assay activity records from ChEMBL."
    )
    parser.add_argument(
        "--terms",
        nargs="+",
        default=list(DEFAULT_TERMS),
        help="Search terms used to discover assays/targets.",
    )
    parser.add_argument(
        "--raw-out",
        default="data/herg_raw.csv",
        help="Path for raw export (CSV/TSV).",
    )
    parser.add_argument(
        "--clean-out",
        default="data/herg_clean.csv",
        help="Path for cleaned export (CSV/TSV).",
    )
    parser.add_argument(
        "--sep",
        default=",",
        help="Output delimiter for both files (default: ','). Use '\\t' for TSV.",
    )
    parser.add_argument(
        "--page-size",
        type=int,
        default=200,
        help="Pagination chunk size for ChEMBL API reads.",
    )
    parser.add_argument(
        "--max-assays",
        type=int,
        default=None,
        help="Optional cap for number of assays to process (debugging).",
    )
    parser.add_argument(
        "--max-activities-per-assay",
        type=int,
        default=None,
        help="Optional cap of activity rows per assay (debugging).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    return parser.parse_args()


def paginated_records(queryset: Any, page_size: int = 200) -> Iterator[Dict[str, Any]]:
    """Iterate paginated ChEMBL records with explicit slicing."""
    offset = 0
    while True:
        page = list(queryset[offset : offset + page_size])
        if not page:
            break
        for item in page:
            yield item
        if len(page) < page_size:
            break
        offset += page_size


def clean_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        normalized = " ".join(value.split())
        return normalized if normalized else None
    return str(value)


def to_key(value: Any) -> str:
    return clean_text(value) or ""


def merge_nonempty(base: Dict[str, Any], incoming: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    for key, value in incoming.items():
        if value not in (None, ""):
            merged[key] = value
    return merged


def discover_target_ids(target_client: Any, terms: Sequence[str], page_size: int) -> Set[str]:
    target_ids: Set[str] = set()

    for term in terms:
        try:
            for target in paginated_records(target_client.search(term), page_size=page_size):
                target_id = target.get("target_chembl_id")
                if target_id:
                    target_ids.add(target_id)
        except Exception as exc:
            logging.warning("Target search failed for term '%s': %s", term, exc)

    # Target annotation filters are attempted explicitly for KCNH2/hERG.
    filter_candidates = [
        {"target_components__gene_symbol__iexact": "KCNH2"},
        {"target_components__component_synonym__iexact": "KCNH2"},
        {"pref_name__icontains": "hERG"},
    ]
    for filter_kwargs in filter_candidates:
        try:
            for target in paginated_records(
                target_client.filter(**filter_kwargs), page_size=page_size
            ):
                target_id = target.get("target_chembl_id")
                if target_id:
                    target_ids.add(target_id)
        except Exception as exc:
            logging.debug("Target filter %s failed: %s", filter_kwargs, exc)

    return target_ids


def discover_assays(
    assay_client: Any,
    target_ids: Iterable[str],
    terms: Sequence[str],
    page_size: int,
) -> Tuple[Set[str], Dict[str, Dict[str, Any]]]:
    assay_ids: Set[str] = set()
    assay_cache: Dict[str, Dict[str, Any]] = {}

    def add_assay(record: Dict[str, Any]) -> None:
        assay_id = record.get("assay_chembl_id")
        if not assay_id:
            return
        assay_ids.add(assay_id)
        assay_cache[assay_id] = merge_nonempty(assay_cache.get(assay_id, {}), record)

    for term in terms:
        try:
            for assay in paginated_records(assay_client.search(term), page_size=page_size):
                add_assay(assay)
        except Exception as exc:
            logging.warning("Assay search failed for term '%s': %s", term, exc)

    for target_id in sorted(target_ids):
        try:
            for assay in paginated_records(
                assay_client.filter(target_chembl_id=target_id), page_size=page_size
            ):
                add_assay(assay)
        except Exception as exc:
            logging.warning(
                "Assay fetch by target failed for %s: %s", to_key(target_id), exc
            )

    return assay_ids, assay_cache


def extract_gene_symbols(component_record: Dict[str, Any]) -> List[str]:
    symbols: Set[str] = set()
    for synonym in component_record.get("target_component_synonyms") or []:
        synonym_name = clean_text(synonym.get("component_synonym"))
        syn_type = (synonym.get("syn_type") or "").upper()
        if not synonym_name:
            continue
        if "GENE" in syn_type:
            symbols.add(synonym_name)
        elif re.fullmatch(r"[A-Z0-9\-]{3,15}", synonym_name):
            # Fallback heuristic for compact all-caps symbols if type is absent.
            symbols.add(synonym_name)
    return sorted(symbols)


def get_target_annotation(
    target_id: Optional[str],
    target_client: Any,
    target_component_client: Any,
    target_cache: Dict[str, Dict[str, Any]],
    component_cache: Dict[int, Dict[str, Any]],
) -> Dict[str, Any]:
    if not target_id:
        return {
            "target_chembl_id": None,
            "target_pref_name": None,
            "target_type": None,
            "target_organism": None,
            "target_components": None,
            "target_gene_symbols": None,
        }

    target_record = target_cache.get(target_id)
    if target_record is None:
        try:
            target_record = target_client.get(target_id) or {}
        except Exception as exc:
            logging.debug("Target lookup failed for %s: %s", target_id, exc)
            target_record = {}
        target_cache[target_id] = target_record

    component_descriptions: List[str] = []
    gene_symbols: Set[str] = set()

    for component in target_record.get("target_components") or []:
        component_id = component.get("component_id")
        accession = clean_text(component.get("accession"))
        component_desc = clean_text(component.get("component_description"))
        label_parts = [part for part in (accession, component_desc) if part]
        if label_parts:
            component_descriptions.append(" | ".join(label_parts))

        if component_id is None:
            continue

        component_record = component_cache.get(component_id)
        if component_record is None:
            try:
                component_record = target_component_client.get(component_id) or {}
            except Exception as exc:
                logging.debug(
                    "Target component lookup failed for %s: %s", component_id, exc
                )
                component_record = {}
            component_cache[component_id] = component_record

        for symbol in extract_gene_symbols(component_record):
            gene_symbols.add(symbol)

    return {
        "target_chembl_id": target_id,
        "target_pref_name": clean_text(target_record.get("pref_name")),
        "target_type": clean_text(target_record.get("target_type")),
        "target_organism": clean_text(target_record.get("organism")),
        "target_components": "; ".join(sorted(set(component_descriptions))) or None,
        "target_gene_symbols": "; ".join(sorted(gene_symbols)) or None,
    }


def get_document_annotation(
    document_chembl_id: Optional[str],
    document_client: Any,
    document_cache: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    if not document_chembl_id:
        return {
            "document_chembl_id": None,
            "document_journal": None,
            "document_year": None,
            "document_title": None,
            "document_doi": None,
            "document_pubmed_id": None,
        }

    doc_record = document_cache.get(document_chembl_id)
    if doc_record is None:
        try:
            doc_record = document_client.get(document_chembl_id) or {}
        except Exception as exc:
            logging.debug(
                "Document lookup failed for %s: %s", document_chembl_id, exc
            )
            doc_record = {}
        document_cache[document_chembl_id] = doc_record

    return {
        "document_chembl_id": document_chembl_id,
        "document_journal": clean_text(doc_record.get("journal")),
        "document_year": doc_record.get("year"),
        "document_title": clean_text(doc_record.get("title")),
        "document_doi": clean_text(doc_record.get("doi")),
        "document_pubmed_id": clean_text(doc_record.get("pubmed_id")),
    }


def get_source_annotation(
    source_id: Optional[Any],
    source_client: Any,
    source_cache: Dict[int, Dict[str, Any]],
) -> Dict[str, Any]:
    if source_id in (None, ""):
        return {
            "source_id": None,
            "source_description": None,
            "source_release": None,
            "source_url": None,
        }

    try:
        source_id_int = int(source_id)
    except Exception:
        source_id_int = None

    if source_id_int is None:
        return {
            "source_id": source_id,
            "source_description": None,
            "source_release": None,
            "source_url": None,
        }

    source_record = source_cache.get(source_id_int)
    if source_record is None:
        try:
            source_record = source_client.get(source_id_int) or {}
        except Exception as exc:
            logging.debug("Source lookup failed for %s: %s", source_id_int, exc)
            source_record = {}
        source_cache[source_id_int] = source_record

    return {
        "source_id": source_id_int,
        "source_description": clean_text(source_record.get("src_description")),
        "source_release": clean_text(source_record.get("src_release")),
        "source_url": clean_text(source_record.get("src_url")),
    }


def get_canonical_smiles(
    molecule_chembl_id: Optional[str],
    molecule_client: Any,
    molecule_cache: Dict[str, Optional[str]],
) -> Optional[str]:
    if not molecule_chembl_id:
        return None
    if molecule_chembl_id in molecule_cache:
        return molecule_cache[molecule_chembl_id]

    smiles: Optional[str] = None
    try:
        molecule_record = molecule_client.get(molecule_chembl_id) or {}
        structures = molecule_record.get("molecule_structures") or {}
        smiles = clean_text(structures.get("canonical_smiles"))
    except Exception as exc:
        logging.debug("Molecule lookup failed for %s: %s", molecule_chembl_id, exc)
    molecule_cache[molecule_chembl_id] = smiles
    return smiles


def build_activity_rows(
    assay_ids: Sequence[str],
    assay_cache: Dict[str, Dict[str, Any]],
    page_size: int,
    max_activities_per_assay: Optional[int],
) -> List[Dict[str, Any]]:
    activity_client = new_client.activity
    target_client = new_client.target
    target_component_client = new_client.target_component
    document_client = new_client.document
    source_client = new_client.source
    molecule_client = new_client.molecule

    target_cache: Dict[str, Dict[str, Any]] = {}
    component_cache: Dict[int, Dict[str, Any]] = {}
    document_cache: Dict[str, Dict[str, Any]] = {}
    source_cache: Dict[int, Dict[str, Any]] = {}
    molecule_cache: Dict[str, Optional[str]] = {}

    rows: List[Dict[str, Any]] = []
    missing_smiles_positions: Dict[str, List[int]] = defaultdict(list)

    total_assays = len(assay_ids)
    for idx, assay_id in enumerate(assay_ids, start=1):
        if idx % 25 == 0 or idx == total_assays:
            logging.info("Processing assay %d/%d (%s)", idx, total_assays, assay_id)

        assay_record = assay_cache.get(assay_id, {})
        target_id_from_assay = clean_text(assay_record.get("target_chembl_id"))

        try:
            queryset = activity_client.filter(assay_chembl_id=assay_id)
            for jdx, activity in enumerate(
                paginated_records(queryset, page_size=page_size), start=1
            ):
                if max_activities_per_assay and jdx > max_activities_per_assay:
                    break

                target_id = clean_text(activity.get("target_chembl_id")) or target_id_from_assay
                source_id = activity.get("src_id", assay_record.get("src_id"))
                document_id = clean_text(
                    activity.get("document_chembl_id")
                    or assay_record.get("document_chembl_id")
                )
                molecule_chembl_id = clean_text(activity.get("molecule_chembl_id"))
                canonical_smiles = clean_text(activity.get("canonical_smiles"))

                row = {
                    "activity_id": clean_text(activity.get("activity_id")),
                    "molecule_chembl_id": molecule_chembl_id,
                    "canonical_smiles": canonical_smiles,
                    "assay_chembl_id": assay_id,
                    "assay_description": clean_text(assay_record.get("description")),
                    "assay_type": clean_text(assay_record.get("assay_type")),
                    "assay_organism": clean_text(assay_record.get("assay_organism")),
                    "assay_cell_type": clean_text(assay_record.get("assay_cell_type")),
                    "assay_tissue": clean_text(assay_record.get("assay_tissue")),
                    "cell_chembl_id": clean_text(assay_record.get("cell_chembl_id")),
                    "standard_type": clean_text(activity.get("standard_type")),
                    "standard_value": clean_text(activity.get("standard_value")),
                    "standard_units": clean_text(activity.get("standard_units")),
                    "standard_relation": clean_text(activity.get("standard_relation")),
                    "activity_comment": clean_text(activity.get("activity_comment")),
                    "pchembl_value": clean_text(activity.get("pchembl_value")),
                    "bao_endpoint": clean_text(activity.get("bao_endpoint")),
                }

                row.update(
                    get_target_annotation(
                        target_id,
                        target_client=target_client,
                        target_component_client=target_component_client,
                        target_cache=target_cache,
                        component_cache=component_cache,
                    )
                )
                row.update(
                    get_document_annotation(
                        document_id,
                        document_client=document_client,
                        document_cache=document_cache,
                    )
                )
                row.update(
                    get_source_annotation(
                        source_id,
                        source_client=source_client,
                        source_cache=source_cache,
                    )
                )

                rows.append(row)
                if not canonical_smiles and molecule_chembl_id:
                    missing_smiles_positions[molecule_chembl_id].append(len(rows) - 1)

        except Exception as exc:
            logging.warning("Activity fetch failed for assay %s: %s", assay_id, exc)

    if missing_smiles_positions:
        logging.info(
            "Backfilling missing canonical SMILES for %d molecules",
            len(missing_smiles_positions),
        )
        for molecule_chembl_id, positions in missing_smiles_positions.items():
            smiles = get_canonical_smiles(
                molecule_chembl_id,
                molecule_client=molecule_client,
                molecule_cache=molecule_cache,
            )
            for pos in positions:
                rows[pos]["canonical_smiles"] = smiles

    return rows


def clean_dataframe(df_raw: pd.DataFrame) -> pd.DataFrame:
    if df_raw.empty:
        return df_raw.copy()

    df = df_raw.copy()
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].apply(clean_text)

    df["standard_value_num"] = pd.to_numeric(df["standard_value"], errors="coerce")
    df["pchembl_value_num"] = pd.to_numeric(df["pchembl_value"], errors="coerce")
    df["document_year"] = pd.to_numeric(df["document_year"], errors="coerce").astype(
        "Int64"
    )

    # Keep rows with core identifiers and at least one quantitative/qualitative endpoint field.
    has_measure = (
        df["standard_type"].notna()
        | df["standard_value"].notna()
        | df["standard_units"].notna()
    )
    df = df[df["assay_chembl_id"].notna() & df["molecule_chembl_id"].notna() & has_measure]

    dedupe_cols = [
        "molecule_chembl_id",
        "canonical_smiles",
        "assay_chembl_id",
        "standard_type",
        "standard_value",
        "standard_units",
        "standard_relation",
        "document_chembl_id",
    ]
    df = df.drop_duplicates(subset=dedupe_cols, keep="first")
    df = df.sort_values(
        by=["assay_chembl_id", "molecule_chembl_id", "standard_type", "standard_value"],
        kind="stable",
    ).reset_index(drop=True)

    # Normalize relation to a strict set where possible.
    allowed_relations = {"=", ">", "<", ">=", "<=", "~"}
    df["standard_relation"] = df["standard_relation"].where(
        df["standard_relation"].isin(allowed_relations), None
    )
    return df


def write_table(df: pd.DataFrame, path: str, sep: str) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    delimiter = "\t" if sep == "\\t" else sep
    df.to_csv(out_path, index=False, sep=delimiter)
    logging.info("Wrote %d rows to %s", len(df), out_path)


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    terms = tuple(dict.fromkeys(t.strip() for t in args.terms if t.strip()))
    if not terms:
        raise ValueError("At least one non-empty search term is required.")

    logging.info("Search terms: %s", ", ".join(terms))
    logging.info("Discovering hERG/KCNH2 targets and assays from ChEMBL...")

    target_client = new_client.target
    assay_client = new_client.assay

    target_ids = discover_target_ids(target_client, terms=terms, page_size=args.page_size)
    logging.info("Discovered %d candidate target IDs", len(target_ids))

    assay_ids, assay_cache = discover_assays(
        assay_client,
        target_ids=target_ids,
        terms=terms,
        page_size=args.page_size,
    )
    sorted_assays = sorted(assay_ids)

    if args.max_assays is not None:
        sorted_assays = sorted_assays[: args.max_assays]
        logging.info("Applying assay cap: %d", len(sorted_assays))
    else:
        logging.info("Discovered %d candidate assays", len(sorted_assays))

    logging.info("Fetching activity records with target and provenance metadata...")
    rows = build_activity_rows(
        assay_ids=sorted_assays,
        assay_cache=assay_cache,
        page_size=args.page_size,
        max_activities_per_assay=args.max_activities_per_assay,
    )

    df_raw = pd.DataFrame(rows)
    if df_raw.empty:
        logging.warning("No activity records found. Writing empty outputs.")

    write_table(df_raw, args.raw_out, sep=args.sep)

    df_clean = clean_dataframe(df_raw)
    write_table(df_clean, args.clean_out, sep=args.sep)

    logging.info(
        "Done. Raw rows: %d | Clean rows: %d | Assays processed: %d",
        len(df_raw),
        len(df_clean),
        len(sorted_assays),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
