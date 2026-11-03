"""Build a clean standalone submission without credentials or starter history."""
from pathlib import Path
import shutil
import zipfile

ROOT = Path(__file__).resolve().parents[1]
FILES = ["main.py", "bandit_utils.py", "gaussian_bandit.py", "experiments.py",
         "pyproject.toml", "uv.lock", ".python-version", "README.md", "PRESENTATION.md"]


def main():
    destination = ROOT / "submission" / "gaussian-bandit"
    destination.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        shutil.copy2(ROOT / name, destination / name)
    for directory in ["pages", "tests", "results", "tools"]:
        shutil.copytree(ROOT / directory, destination / directory, dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    workflow = destination / ".github" / "workflows"
    workflow.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / ".github/workflows/frontend-tests.yml", workflow)
    (destination / ".gitignore").write_text(
        ".venv/\n__pycache__/\n*.py[cod]\n.env\n.streamlit/secrets.toml\n"
        "*.egg-info/\nbuild/\ndist/\nsubmission/\n.DS_Store\n")
    # Whitelist archive contents even if this export has since been initialized
    # as a Git repository or used with its own virtual environment.
    candidates = [destination / name for name in FILES] + [destination / ".gitignore"]
    candidates.append(workflow / "frontend-tests.yml")
    for directory in ["pages", "tests", "results", "tools"]:
        candidates.extend((destination / directory).rglob("*"))
    with zipfile.ZipFile(destination.with_suffix(".zip"), "w", zipfile.ZIP_DEFLATED) as archive:
        for path in candidates:
            if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc":
                archive.write(path, path.relative_to(destination))
    print(destination)
    print(destination.with_suffix(".zip"))


if __name__ == "__main__":
    main()
