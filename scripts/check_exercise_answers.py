"""Run embedded guided examples and Practice 1 checks without dataset downloads.

Full real-data exercise coverage is provided by execute_notebooks.py.
Student attempt cells are intentionally skipped; the reference answers are
executed from the notebook itself, with no external answer-guide dependency.
"""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import balanced_accuracy_score
import nbformat

root = Path(__file__).resolve().parents[1]
for path in sorted((root / 'notebooks').glob('*.ipynb')):
    nb = nbformat.read(path, as_version=4)
    env = dict(np=np, pd=pd, plt=plt, signal=signal,
               StandardScaler=StandardScaler, GroupKFold=GroupKFold,
               balanced_accuracy_score=balanced_accuracy_score,
               rng=np.random.default_rng(613))
    checks = 0
    for cell in nb.cells:
        if cell.cell_type == 'markdown' and cell.source.startswith('## Apply the ideas'):
            break
        if cell.cell_type != 'code' or cell.source.startswith('# Colab:'):
            continue
        if 'return None' in cell.source or cell.source.startswith('# Your attempt'):
            continue
        exec(compile(cell.source, path.name, 'exec'), env)
        if 'Exercise pending:' in cell.source:
            assert env['answer'] is not None, path.name
            checks += 1
    assert checks == 1, (path.name, checks)
    plt.close('all')
    print('PASS', path.name)
print('All 16 embedded reference checks passed.')
