import hashlib
import subprocess
import sys
from pathlib import Path

from necro.training import trainer


def test_training_help_runs_without_loading_training_dependencies():
    script = """
import runpy
import sys

class RejectTrainingDependencies:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'peft', 'transformers'}:
            raise AssertionError(f'Unexpected training dependency: {fullname}')

sys.meta_path.insert(0, RejectTrainingDependencies())
sys.argv = ['necro.training', '--help']
runpy.run_module('necro.training', run_name='__main__')
"""
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=True, timeout=30
    )
    assert "--initial-adapter" in result.stdout
    assert "--objective {answer-ce,candidate-ce}" in result.stdout


def test_source_snapshot_resolves_relocated_code_outside_project_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    hashes = trainer.snapshot_sources(tmp_path)
    package = Path(trainer.__file__).resolve().parents[1]
    assert "training/trainer.py" in hashes
    assert "training/__main__.py" in hashes
    assert "engine.py" in hashes
    for name, digest in hashes.items():
        saved = tmp_path / "source" / name
        assert saved.read_bytes() == (package / name).read_bytes()
        assert hashlib.sha256(saved.read_bytes()).hexdigest() == digest
