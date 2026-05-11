from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PRETRAINED_DIR = PROJECT_ROOT / "models" / "pretrained"


def project_path(*parts):
    return PROJECT_ROOT.joinpath(*parts)


def resolve_pretrained(name):
    local_weight = PRETRAINED_DIR / name
    return str(local_weight) if local_weight.is_file() else name
