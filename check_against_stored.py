#!/usr/bin/env python3
"""Compare the outputs in results/, tex/ and figures/ with the stored outputs of the article.

STORED_OUTPUTS_SHA256.txt lists, for every output distributed in this archive, the SHA-256
of the file (CSV, JSON, text, LaTeX table bodies and number macros) or, for a figure, of its
first page rendered at 100 dpi (a PDF contains its creation date, so the files themselves
always differ).

    python check_against_stored.py           compare; exit status 1 if a numerical output differs
    python check_against_stored.py --write   rewrite the list from the present files
                                             (only after a deliberate change of the results)

A difference in a numerical output after a rerun with the package versions of
requirements-lock.txt means that the stored result was not reproduced. With other versions
the last digits may differ; compare the file with the copy in the archive. Differences in
figures are listed but do not change the exit status.
"""
import hashlib, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LIST = ROOT / 'STORED_OUTPUTS_SHA256.txt'
TEXT = ('.csv', '.json', '.txt', '.tex')
SKIP = ('logs',)          # run logs contain elapsed times


def outputs():
    files = [f for d in ('results', 'tex') for f in sorted((ROOT / d).rglob('*'))
             if f.is_file() and f.suffix in TEXT and not set(SKIP) & set(f.relative_to(ROOT).parts)]
    files += [ROOT / 'PYTHON_VALIDATION_REPORT.json']
    files += [f for f in sorted((ROOT / 'figures').glob('*.pdf'))]
    return files


def digest(f):
    if f.suffix == '.pdf':
        try:
            import pymupdf
        except ImportError:                  # PyMuPDF before 1.24.3
            import fitz as pymupdf
        pix = pymupdf.open(f)[0].get_pixmap(dpi=100, alpha=False)
        return hashlib.sha256(bytes(pix.samples)).hexdigest()
    return hashlib.sha256(f.read_bytes()).hexdigest()


def main():
    if '--write' in sys.argv[1:]:
        LIST.write_text(''.join(f'{digest(f)}  {f.relative_to(ROOT).as_posix()}\n' for f in outputs()))
        print('Wrote', LIST.name, 'for', len(outputs()), 'outputs')
        return 0
    stored = dict(reversed(line.split('  ', 1)) for line in LIST.read_text().splitlines() if line.strip())
    present = {f.relative_to(ROOT).as_posix(): f for f in outputs()}
    same, numeric, figures = 0, [], []
    for name, h in stored.items():
        f = present.get(name)
        if f is not None and digest(f) == h:
            same += 1
        elif name.endswith('.pdf'):
            figures.append(name + ('' if f is not None else ' (missing)'))
        else:
            numeric.append(name + ('' if f is not None else ' (missing)'))
    new = sorted(set(present) - set(stored))
    print(f'{same} of {len(stored)} stored outputs are identical')
    for label, names in (('numerical outputs that differ', numeric), ('figures that differ when rendered', figures),
                         ('outputs not in the stored list', new)):
        if names:
            print(f'{len(names)} {label}:')
            for n in names:
                print('   ', n)
    if not numeric:
        print('PASS: every numerical output equals the stored output of the article')
    return 1 if numeric else 0


if __name__ == '__main__':
    sys.exit(main())
