#!/usr/bin/env python3
"""TIFF copies (300 dpi, LZW) of the figure PDFs that the two documents use.

The PDFs written by the plotting scripts are the authoritative vector figures; the TIFF
copies are rendered from them, so that the two cannot differ. Stale TIFF files of figures
that are no longer used are removed.
"""
import io, re
from pathlib import Path
import fitz
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / 'figures'
used = set()
for doc in ('sec_tables_figures.tex', 'supplement.tex'):
    used |= set(re.findall(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\.pdf\}', (ROOT / 'tex' / doc).read_text()))
for old in list(FIG.glob('*.tif')) + list(FIG.glob('*.png')):
    if old.stem not in used or old.suffix == '.png':
        old.unlink()
for stem in sorted(used):
    page = fitz.open(FIG / f'{stem}.pdf')[0]
    pix = page.get_pixmap(dpi=300, alpha=False)
    Image.open(io.BytesIO(pix.tobytes('png'))).convert('RGB').save(FIG / f'{stem}.tif', compression='tiff_lzw', dpi=(300, 300))
print('TIFF copies written for', len(used), 'figures:', ', '.join(sorted(used)))
