"""Shared artifact, environment, and path utilities."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
from importlib.metadata import version
from gene_complexity._paths import ROOT, CONFIG
from gene_complexity.preprocessing.classical import digest

def config_arg(p):
    p.add_argument('--config', type=Path, default=CONFIG, help='Input-artifact JSON config')

def read(path):
    return json.loads(Path(path).read_text())

def write(path, data):
    Path(path).write_text(json.dumps(data, indent=2)+'\n')

def fresh(path, prefix):
    path = Path(path) if path else ROOT/'results/workflow'/f'{prefix}_{datetime.now(timezone.utc):%Y%m%dT%H%M%S_%fZ}'
    path.mkdir(parents=True, exist_ok=False)
    return path

def environment():
    return dict(python=platform.python_version(), versions={p:version(p) for p in ['numpy','scipy','scikit-learn','pandas']})

