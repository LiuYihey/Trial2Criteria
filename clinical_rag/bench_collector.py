"""
Collect four-domain RAG profiles from final_bench.json.

Output format matches trial_profile_*.json:
  disease_info, drug_info, relevant_papers, similar_trials
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys

from tqdm import tqdm as _tqdm

script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.abspath(os.path.join(script_dir, ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from clinical_rag.batch import EnhancedMedicalSearchLite
from clinical_rag.config import load_config, resolve_path
from clinical_rag.paths import setup_paths
from clinical_rag.utils import make_json_serializable

setup_paths(_project_root)

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


# ---- tqdm wrapper: use ASCII bar in PowerShell to avoid Unicode block garble ----
_is_powershell = bool(os.environ.get("PSModulePath"))

def tqdm(iterable, **kwargs):
    if _is_powershell:
        kwargs.setdefault("ascii", True)
    kwargs.setdefault("file", sys.stdout)   # avoid PowerShell red stderr
    return _tqdm(iterable, **kwargs)


def profile_filename(title: str, max_len: int = 50) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", (title or "untitled").strip())
    slug = re.sub(r"_+", "_", slug).strip("_")
    if len(slug) > max_len:
        slug = slug[:max_len]
    return f"trial_profile_{slug}.json"


def rag_result_to_profile(rag_result: dict) -> dict:
    """Convert internal batch RAG result to trial_profile JSON format."""
    similar_parts = []
    for phase_num in (1, 2, 3):
        phase_text = rag_result.get(f"Phase{phase_num}_clinical_trials")
        if phase_text and phase_text != "none":
            similar_parts.append(f"--- Similar Phase{phase_num} Trials ---\n{phase_text}")

    disease = rag_result.get("Disease_info") or "none"
    drug = rag_result.get("Drugbank_info") or "none"
    papers = rag_result.get("Papers_info") or "none"
    similar = "\n\n".join(similar_parts) if similar_parts else "none"

    return {
        "disease_info": disease,
        "drug_info": drug,
        "relevant_papers": papers,
        "similar_trials": similar,
    }


def load_manifest(manifest_path: str) -> dict:
    if not os.path.exists(manifest_path):
        return {}
    with open(manifest_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_manifest(manifest_path: str, manifest: dict) -> None:
    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)


def unique_output_path(output_dir: str, title: str, nct_id: str) -> str:
    # Always use NCT ID as filename for consistency
    return os.path.join(output_dir, f"{nct_id}.json")


def collect_profiles(
    input_json: str,
    output_dir: str,
    nct_id: str | None = None,
    limit: int | None = None,
    force: bool = False,
) -> None:
    input_path = resolve_path(input_json, _project_root)
    output_path = resolve_path(output_dir, _project_root)
    os.makedirs(output_path, exist_ok=True)

    manifest_path = os.path.join(output_path, "manifest.json")
    manifest = load_manifest(manifest_path)

    with open(input_path, "r", encoding="utf-8") as f:
        trials = json.load(f)

    # ---- Deduplicate by NCT ID (keep first occurrence) ----
    seen_nct: set[str] = set()
    unique_trials = []
    for t in trials:
        nid = t.get("protocolSection", {}).get("identificationModule", {}).get("nctId")
        if nid and nid in seen_nct:
            continue
        if nid:
            seen_nct.add(nid)
        unique_trials.append(t)
    if len(unique_trials) < len(trials):
        print(f"--- [INFO] Deduplicated: {len(trials)} -> {len(unique_trials)} trials (removed {len(trials) - len(unique_trials)} duplicates) ---")
    trials = unique_trials

    if nct_id:
        trials = [
            t
            for t in trials
            if t.get("protocolSection", {}).get("identificationModule", {}).get("nctId") == nct_id
        ]
        if not trials:
            print(f"--- [FATAL] NCT ID '{nct_id}' not found in {input_path} ---")
            return

    if limit is not None:
        trials = trials[:limit]

    pending = trials if force else [
        t
        for t in trials
        if t.get("protocolSection", {}).get("identificationModule", {}).get("nctId") not in manifest
    ]
    print(f"--- [INFO] Total: {len(trials)}, pending: {len(pending)}, output: {output_path} ---")

    if not pending:
        print("--- [INFO] All trials already processed. Use --force to regenerate. ---")
        return

    config = load_config()
    print("--- [INFO] Creating search system (lazy mode, models load on demand)... ---", flush=True)
    search_system = EnhancedMedicalSearchLite(config)

    print("--- [INFO] Checking trial retrieval data (first access triggers loading)... ---", flush=True)
    loaded_phases = list(getattr(search_system, "trial_data_by_phase", {}) or {})
    if not loaded_phases:
        print("--- [FATAL] No trial retrieval data loaded. ---")
        print("--- [FATAL] Required files under data/trial_gov/: ---")
        print("  embeddings/Phase{1,2,3}_emb_only_fields.pth")
        print("  retrieval_base/RAG_base_phase{1,2,3}.csv")
        return
    print(f"--- [INFO] Similar-trial retrieval ready for phases: {loaded_phases} ---")

    for record in tqdm(pending, desc="Collecting RAG profiles"):
        nct = record.get("protocolSection", {}).get("identificationModule", {}).get("nctId", "UNKNOWN")
        title = record.get("protocolSection", {}).get("identificationModule", {}).get("officialTitle", "")
        print(f"\n{'=' * 20} {nct}: {title[:80]} {'=' * 20}")

        try:
            rag_result, _ = search_system.process_bench_record(record)
            profile = rag_result_to_profile(rag_result)
            out_file = unique_output_path(output_path, title, nct)

            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(make_json_serializable(profile), f, indent=2, ensure_ascii=False)

            manifest[nct] = {
                "file": os.path.basename(out_file),
                "title": title,
            }
            save_manifest(manifest_path, manifest)
            print(f"--- [SUCCESS] Saved {out_file} ---")
        except Exception as exc:
            print(f"--- [ERROR] Failed on {nct}: {exc} ---")
            import traceback
            traceback.print_exc()

    print(f"\n--- [DONE] Profiles in {output_path} ({len(manifest)} total in manifest) ---")


def main():
    parser = argparse.ArgumentParser(
        description="Collect four-domain RAG data from final_bench.json into trial_profile_*.json files."
    )
    parser.add_argument(
        "--input",
        default="final_bench.json",
        help="Path to final_bench.json (ClinicalTrials.gov JSON array).",
    )
    parser.add_argument(
        "--output_dir",
        default="outputs/bench_profiles",
        help="Directory for trial_profile_*.json outputs.",
    )
    parser.add_argument("--nct_id", default=None, help="Process a single NCT ID only.")
    parser.add_argument("--limit", type=int, default=None, help="Process at most N trials (for testing).")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate profiles even if NCT ID is already in manifest.",
    )
    args = parser.parse_args()

    collect_profiles(
        input_json=args.input,
        output_dir=args.output_dir,
        nct_id=args.nct_id,
        limit=args.limit,
        force=args.force,
    )


if __name__ == "__main__":
    main()
