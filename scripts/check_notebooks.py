"""Validate structure, syntax, source synchronization and local Markdown links."""
from pathlib import Path
import ast,re
import nbformat
from build_notebooks import source_cells,ROOT
count=0
for path in sorted((ROOT/'notebooks').glob('*.ipynb')):
    nb=nbformat.read(path,as_version=4); nbformat.validate(nb)
    expected=source_cells(ROOT/'notebook_src'/f'{path.stem}.py')
    assert [(c.cell_type,c.source.strip()) for c in nb.cells] == [(c.cell_type,c.source.strip()) for c in expected],path
    for i,cell in enumerate(nb.cells):
        if cell.cell_type=='code': ast.parse(cell.source,filename=f'{path.name}:{i}')
        assert not any(o.output_type=='error' for o in cell.get('outputs',[])),path
    count+=1
for path in ROOT.rglob('*.md'):
    for target in re.findall(r'\]\(([^)]+)\)',path.read_text()):
        if '://' in target or target.startswith('#'): continue
        assert (path.parent/target.split('#')[0]).exists(),(path,target)
print(f'{count} notebooks: valid, parseable, source-synchronized; local Markdown links valid.')
