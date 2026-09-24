"""Check the small instructor functions without downloading any datasets.

Requires the core course environment. Open-ended exercise investigations are
assessed using the discussion criteria, not a fixed target accuracy.
"""
from pathlib import Path
import re
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
guide = (root / 'docs/guides/instructor-solutions.md').read_text()
sections = dict(re.findall(r'^## (\d\d_\w+)\n(.*?)(?=^## |\Z)', guide, re.M | re.S))
for path in sorted((root / 'notebooks').glob('*.ipynb')):
    nb = nbformat.read(path, as_version=4)
    env = dict(np=np, pd=pd, plt=plt, signal=signal,
               StandardScaler=StandardScaler, GroupKFold=GroupKFold,
               balanced_accuracy_score=balanced_accuracy_score,
               rng=np.random.default_rng(613))
    for cell in nb.cells:
        if cell.cell_type == 'markdown' and cell.source.startswith('## Apply the ideas'):
            break
        if cell.cell_type == 'code' and not cell.source.startswith('# Colab:'):
            exec(compile(cell.source, path.name, 'exec'), env)
    solution = re.search(r'```python\n(.*?)\n```', sections[path.stem], re.S).group(1)
    exec(compile(solution, path.name + ':reference', 'exec'), env)
    check = next(c.source for c in nb.cells if c.cell_type == 'code' and 'Exercise pending:' in c.source)
    exec(compile(check, path.name + ':check', 'exec'), env)
    assert env['answer'] is not None, path.name
    plt.close('all')
    print('PASS', path.name)
print('All 16 reference functions passed their stated checks.')
