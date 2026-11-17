#!/usr/bin/env python3
"""TIFF copies (300 dpi, LZW) of the figure PDFs of the article.

The PDFs written by the plotting scripts are the authoritative vector figures; the TIFF
copies are rendered from them, so that the two cannot differ. Every PDF in figures/ is a
figure of the article (six in the main text, seven in S1 Appendix). Stale TIFF and PNG
files are removed.

figures/journal/Fig1.tif to Fig6.tif are the six main-text figures under the file names of
the journal and within its width limit (2250 pixels at 300 dpi): each is rendered from its
PDF at 300 dpi, or at the largest scale that keeps the width within 2200 pixels.
"""
import io
from pathlib import Path
import fitz
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / 'figures'
MAIN = ('fig1_model', 'fig2_por_by_onset', 'fig4_bias', 'fig5_precision', 'fig_nordcan', 'fig_colon_bounds')
MAXWIDTH = 2200


def save(pix, dst):
    Image.open(io.BytesIO(pix.tobytes('png'))).convert('RGB').save(dst, compression='tiff_lzw', dpi=(300, 300))


used = {f.stem for f in FIG.glob('*.pdf')}
assert set(MAIN) <= used, sorted(set(MAIN) - used)
for old in list(FIG.glob('*.tif')) + list(FIG.glob('*.png')):
    if old.stem not in used or old.suffix == '.png':
        old.unlink()
for stem in sorted(used):
    page = fitz.open(FIG / f'{stem}.pdf')[0]
    save(page.get_pixmap(dpi=300, alpha=False), FIG / f'{stem}.tif')
(FIG / 'journal').mkdir(exist_ok=True)
for k, stem in enumerate(MAIN, 1):
    page = fitz.open(FIG / f'{stem}.pdf')[0]
    scale = min(300 / 72, MAXWIDTH / page.rect.width)
    save(page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False), FIG / 'journal' / f'Fig{k}.tif')
print('TIFF copies written for', len(used), 'figures:', ', '.join(sorted(used)))
print('Journal copies of the main-text figures: figures/journal/Fig1.tif to Fig%d.tif' % len(MAIN))
