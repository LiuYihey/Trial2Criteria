"""Four-domain clinical trial RAG: PrimeKG, DrugBank, ClinicalTrials.gov, PubMed."""

from clinical_rag.config import load_config, resolve_path

# Lazy import to avoid loading heavy search.py dependencies on package import
def __getattr__(name):
    if name == "EnhancedMedicalSearch":
        from clinical_rag.search import EnhancedMedicalSearch
        return EnhancedMedicalSearch
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ["EnhancedMedicalSearch", "load_config", "resolve_path"]

__version__ = "0.1.0"
