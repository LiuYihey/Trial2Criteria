import os

from Bio import Entrez


def configure_entrez(config: dict) -> str:
    """Configure NCBI Entrez using email only (no API key required)."""
    email = config.get("api_keys", {}).get("entrez_email", "").strip()
    if not email:
        raise ValueError("'entrez_email' is required in config.json api_keys (or set ENTREZ_EMAIL)")
    os.environ["Entrez.email"] = email
    Entrez.email = email
    return email
