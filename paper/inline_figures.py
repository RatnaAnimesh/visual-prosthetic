"""Produce a self-contained copy of manuscript.html.

paper/manuscript.html links its figures relatively, which is what you want
when reading it out of the repository.  Anywhere else -- an email attachment,
a hosted page under a strict content policy -- the links break.  This rewrites
each <img src="figures/figN.png"> as an inline data URI so the file stands
alone.

Run:  python3 paper/inline_figures.py [output.html]
"""

from __future__ import annotations

import base64
import pathlib
import re
import sys

HERE = pathlib.Path(__file__).resolve().parent
SRC = HERE / "manuscript.html"


def inline(html: str) -> str:
    def sub(m: re.Match) -> str:
        path = HERE / m.group(1)
        if not path.exists():
            raise FileNotFoundError(f"figure referenced but missing: {path}")
        b64 = base64.b64encode(path.read_bytes()).decode("ascii")
        return f'src="data:image/png;base64,{b64}"'

    out, n = re.subn(r'src="(figures/[^"]+\.png)"', sub, html)
    if not n:
        raise RuntimeError("no figure references found -- did the markup change?")
    print(f"inlined {n} figures")
    return out


def main() -> None:
    dest = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else \
        HERE / "manuscript_standalone.html"
    dest.write_text(inline(SRC.read_text()))
    print(f"wrote {dest}  ({dest.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
