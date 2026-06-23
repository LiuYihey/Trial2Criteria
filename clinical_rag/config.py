import glob
import json
import os


def get_project_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def load_config(config_path: str | None = None) -> dict:
    """Load config.json from the project root (or an explicit path)."""
    root = get_project_root()
    path = config_path or os.path.join(root, "config.json")
    try:
        with open(path, "r", encoding="utf-8") as f:
            config = json.load(f)
    except Exception as exc:
        print(f"--- [ERROR] Error loading {path}: {exc} ---")
        raise SystemExit(1) from exc

    _apply_env_overrides(config)
    _merge_api_key_md(config, root)
    return config


def _merge_api_key_md(config: dict, root: str) -> None:
    """Load LLM settings from api_key.md when present (overrides empty llm fields)."""
    path = os.path.join(root, "api_key.md")
    if not os.path.exists(path):
        return

    md_values: dict[str, str] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("ANTHROPIC_BASE_URL="):
                md_values["base_url"] = line.split("=", 1)[1].strip()
            elif line.upper().startswith("API KEY:"):
                md_values["api_key"] = line.split(":", 1)[1].strip()
            elif line.upper().startswith("MODEL:"):
                md_values["model"] = line.split(":", 1)[1].strip()

    llm = config.setdefault("llm", {})
    for key, value in md_values.items():
        if value:
            llm[key] = value


def _apply_env_overrides(config: dict) -> None:
    """Allow secrets to be supplied via environment variables."""
    api_keys = config.setdefault("api_keys", {})
    llm = config.setdefault("llm", {})
    if os.environ.get("ENTREZ_EMAIL"):
        api_keys["entrez_email"] = os.environ["ENTREZ_EMAIL"]
    if os.environ.get("LLM_API_KEY"):
        llm["api_key"] = os.environ["LLM_API_KEY"]
    if os.environ.get("ANTHROPIC_BASE_URL"):
        llm["base_url"] = os.environ["ANTHROPIC_BASE_URL"]
    if os.environ.get("LLM_MODEL"):
        llm["model"] = os.environ["LLM_MODEL"]


def resolve_path(path_str: str, project_root: str | None = None) -> str:
    """Resolve a config-relative path against the project root."""
    if not path_str:
        return path_str

    root = project_root or get_project_root()
    if os.path.isabs(path_str) and os.path.exists(path_str):
        return os.path.abspath(path_str)

    joined = os.path.join(root, path_str.replace("/", os.sep))
    if os.path.exists(joined):
        return os.path.abspath(joined)

    if os.path.exists(path_str):
        return os.path.abspath(path_str)

    legacy_prefixes = (
        "Pubmed/",
        f"Pubmed{os.sep}",
        "self/trial_gov/",
        f"self{os.sep}trial_gov{os.sep}",
        "PrimeKG-main/",
        f"PrimeKG-main{os.sep}",
        "RAG_context_batch_generation/",
        f"RAG_context_batch_generation{os.sep}",
        "Prompt_template/",
        f"Prompt_template{os.sep}",
    )
    candidates = [os.path.join(root, path_str)]
    for prefix in legacy_prefixes:
        if path_str.startswith(prefix):
            candidates.append(os.path.join(root, path_str[len(prefix):]))

    for candidate in candidates:
        if os.path.exists(candidate):
            return os.path.abspath(candidate)

    basename = os.path.basename(path_str)
    if basename:
        matches = glob.glob(os.path.join(root, "**", basename), recursive=True)
        if matches:
            return os.path.abspath(matches[0])

    return os.path.abspath(joined) if os.path.dirname(joined) else path_str
