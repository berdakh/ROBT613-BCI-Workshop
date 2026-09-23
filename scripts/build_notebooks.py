"""Build readable percent-format teaching sources with nbformat.

No code executes here. Existing notebook outputs are intentionally cleared.
"""
from pathlib import Path
import nbformat
ROOT = Path(__file__).resolve().parents[1]

def source_cells(path):
    cells, lines, kind = [], [], None
    def flush():
        if kind is None:
            return
        text = '\n'.join(lines).strip()
        if kind == 'markdown':
            text = '\n'.join(line[2:] if line.startswith('# ') else line[1:] if line == '#' else line
                             for line in text.splitlines()).strip()
            cells.append(nbformat.v4.new_markdown_cell(text))
        else:
            cells.append(nbformat.v4.new_code_cell(text))
    for line in path.read_text().splitlines():
        if line.startswith('# %%'):
            flush(); lines=[]
            kind='markdown' if '[markdown]' in line else 'code'
        else:
            lines.append(line)
    flush()
    return cells

if __name__ == '__main__':
    for source in sorted((ROOT/'notebook_src').glob('*.py')):
        nb=nbformat.v4.new_notebook(cells=source_cells(source),metadata={
            'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
            'language_info':{'name':'python'},
            'colab':{'name':source.stem+'.ipynb','provenance':[]}})
        nbformat.write(nb,ROOT/'notebooks'/f'{source.stem}.ipynb')
        print(source.stem)
