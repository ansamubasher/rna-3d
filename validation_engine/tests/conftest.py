"""Shared pytest setup: make the modules in src/ importable as plain modules
(prepare, rank, evaluate, ...), exactly as the scripts import each other."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
