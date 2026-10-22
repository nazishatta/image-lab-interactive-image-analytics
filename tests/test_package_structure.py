"""Lightweight checks that do not require launching Streamlit."""
from pathlib import Path
import ast

ROOT = Path(__file__).resolve().parents[1]


def test_app_is_parseable():
    ast.parse((ROOT / "app.py").read_text(encoding="utf-8"))


def test_streamlit_deployment_files_exist():
    assert (ROOT / "requirements.txt").is_file()
    assert (ROOT / ".streamlit" / "config.toml").is_file()


def test_unwanted_instructional_ui_absent():
    text = (ROOT / "app.py").read_text(encoding="utf-8")
    for phrase in ("Before Blackboard submission", "Learning scaffold", "Rubric"):
        assert phrase not in text
