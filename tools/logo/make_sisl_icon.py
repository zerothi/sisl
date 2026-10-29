# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""Build the sisl logo (SVG).

"sisl" drawn as atoms and bonds of hexagons, between the bonding orbital (light
blue) and the double-well potential with its two atoms (dark blue) of a dimer.
Each "s" is a one-hexagon wide chain, "i" and "l" are stacks of rotated hexagons.
The 32 px icon shows only the letter "s" of the Orbitron font.

Usage: python make_sisl_icon.py Orbitron.ttf [OUTDIR]
Writes sisl-icon-{light,dark}-{32,512}.svg to OUTDIR (default: the current directory):
"light" is for white backgrounds, "dark" for dark backgrounds.
Needs fontTools. See techs/dev-06-public-docs.md#logo.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

from fontTools.pens.boundsPen import BoundsPen
from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.ttLib import TTFont

# letter colour, then pastel blues (orbital, potential), per background
COLORS = {
    "light": ("#24406b", "#5796de", "#4f78b0"),
    "dark": ("#e8f1ff", "#a8cdf5", "#6c92c9"),
}

LICENSE = (
    "<!-- sisl logo. This file is subject to the terms of the Mozilla Public License, v. 2.0."
    " You can obtain one at https://mozilla.org/MPL/2.0/. -->"
)
FONT = (
    '<!-- The letter "s" is an outline of the Orbitron font by Matt McInerney,'
    " licensed under the SIL Open Font License, Version 1.1. -->"
)

# all drawing is on a 512 grid, scaled to the icon size
SITES = (176, 336)
X0, X1 = 40, 472
B = 9  # bond length of the hexagons in the letters
A = math.sqrt(3) * B


def phi(x, s):
    return 1 / math.cosh((x - s) / 45)


def _bond(x):
    return 155 - 70 * (phi(x, SITES[0]) + phi(x, SITES[1])) / 1.3


def _potential(x):
    return 285 + 90 * sum(26 / math.hypot(x - s, 26) for s in SITES)


# stretch both curves vertically so the drawing fills the square: the top of the
# orbital stroke and the bottom of the 512 px potential atoms sit 30 from the edge, as the sides
_TOP, _BOT = min(_bond(X0 + i) for i in range(X1 - X0 + 1)), _potential(SITES[0])


def _stretch(y):
    return 30 + 9.6 + (y - _TOP) * (512 - 30 - 28 - 30 - 9.6) / (_BOT - _TOP)


def bond(x):
    """Bonding orbital of the two sites."""
    return _stretch(_bond(x))


def potential(x):
    """Soft-Coulomb double well, pointing down in SVG coordinates."""
    return _stretch(_potential(x))


def curve(f, x0, x1, n=240):
    return "M" + " L".join(
        f"{x:.2f},{f(x):.2f}" for x in (x0 + (x1 - x0) * i / n for i in range(n + 1))
    )


def stroke(d, col, w):
    return (
        f'<path d="{d}" fill="none" stroke="{col}" stroke-width="{w}" '
        f'stroke-linecap="round" stroke-linejoin="round"/>'
    )


def glyph(font, ch, cx, cy, width, ink):
    """Glyph scaled so its ink spans width, centred on its ink and the x-height."""
    f = TTFont(font)
    gs = f.getGlyphSet(location={"wght": 900})
    cmap = f.getBestCmap()

    def bounds(c):
        b = BoundsPen(gs)
        gs[cmap[ord(c)]].draw(b)
        return b.bounds

    p = SVGPathPen(gs)
    gs[cmap[ord(ch)]].draw(p)
    x0, _, x1, _ = bounds(ch)
    s = width / (x1 - x0)
    tx, ty = cx - (x0 + x1) / 2 * s, cy + bounds("x")[3] / 2 * s
    return f'<path transform="translate({tx:.2f},{ty:.2f}) scale({s:.5f},{-s:.5f})" d="{p.getCommands()}" fill="{ink}"/>'


def near(p, q, d):
    return abs(math.dist(p, q) - d) < 0.05 * d


def hexagons(centres, t0):
    """Atoms of hexagons (first vertex at angle t0 degrees), shared atoms once."""
    out = []
    for hx, hy in centres:
        for k in range(6):
            t = math.radians(t0 + 60 * k)
            p = (hx + B * math.cos(t), hy + B * math.sin(t))
            if all(math.dist(p, q) > 0.05 * B for q in out):
                out.append(p)
    return out


def chain_s(cx, cy, half=2):
    """A one-hexagon wide "s" of 7 rows of pointy-top hexagons around (cx, cy).
    Rows from top: bar, left curve (2 rows), middle bar of 2*half+1 hexagons,
    right curve (2 rows), bar; invariant under a 180 degree rotation."""
    L, R = -half, half
    rows = [
        [L + 0.5 + i for i in range(R - L + 1)],
        [L],
        [L - 0.5],
        list(range(L, R + 1)),
        [R + 0.5],
        [R],
        [L - 0.5 + i for i in range(R - L + 1)],
    ]
    return hexagons(
        [
            (cx + c * A, cy + (r - 3) * 1.5 * B)
            for r, row in enumerate(rows)
            for c in row
        ],
        30,
    )


def stick(n, foot=False):
    """A vertical stack of n flat-top hexagons, with an optional foot (bottom right);
    the atoms span x >= 0 and end at y = 0."""
    c = [(0.0, -k * A) for k in range(n)]
    if foot:
        c.append((1.5 * B, A / 2))
    pts = hexagons(c, 0)
    x0, y1 = min(x for x, _ in pts), max(y for _, y in pts)
    return [(x - x0, y - y1) for x, y in pts]


def fit(pts, cx, cy, dy, margin=8):
    """Largest scale about (cx, cy), then shifted by dy, that keeps the atoms inside
    the band, margin away from the orbital (stroke half width 9.6) and the
    potential (half width 8)."""

    def ok(f):
        r = 0.38 * B * f
        for x, y in pts:
            x, y = cx + f * (x - cx), cy + dy + f * (y - cy)
            if not (
                X0 + 10 < x < X1 - 10
                and bond(x) + 9.6 + margin < y - r
                and y + r < potential(x) - 8 - margin
            ):
                return False
        return True

    lo, hi = 0.5, 2.0
    for _ in range(30):
        lo, hi = ((lo + hi) / 2, hi) if ok((lo + hi) / 2) else (lo, (lo + hi) / 2)
    return lo


def imbalance(pts, cx, cy, dy, f):
    """Difference between the clearances to the orbital and to the potential."""
    r = 0.38 * B * f
    moved = [(cx + f * (x - cx), cy + dy + f * (y - cy)) for x, y in pts]
    top = min(y - r - bond(x) for x, y in moved)
    bot = min(potential(x) - y - r for x, y in moved)
    return abs(top - bot)


def letters(ink):
    """Atoms and bonds of "sisl", as large as fits between the curves."""
    s = chain_s(136.399, 247.5)
    s2 = [(x + 11 * A, y) for x, y in s]
    # "i" and "l" on the baseline of the s, with even gaps
    i, l = stick(6), stick(7, foot=True)
    base = max(y for _, y in s)
    s1R, s2L, s2R = max(x for x, _ in s), min(x for x, _ in s2), max(x for x, _ in s2)
    gap = (s2L - s1R - max(x for x, _ in i)) / 2
    place = lambda pts, x0: [(x + x0, y + base) for x, y in pts]
    pts = s + place(i, s1R + gap) + s2 + place(l, s2R + gap)
    bonds = [(p, q) for k, p in enumerate(pts) for q in pts[k + 1 :] if near(p, q, B)]
    # scale about the word centre to fill the band, clear of both curves
    cx = (min(x for x, _ in pts) + max(x for x, _ in pts)) / 2
    cy = (min(y for _, y in pts) + max(y for _, y in pts)) / 2
    # largest scale, then the shift with equal clearance above and below
    fits = {dy: fit(pts, cx, cy, dy) for dy in range(-80, 81, 2)}
    f = max(fits.values())
    dy = min(
        (dy for dy in fits if fits[dy] > f - 1e-6),
        key=lambda dy: imbalance(pts, cx, cy, dy, f),
    )
    tr = lambda p: (cx + f * (p[0] - cx), cy + dy + f * (p[1] - cy))
    pts, bonds, b = [tr(p) for p in pts], [(tr(p), tr(q)) for p, q in bonds], B * f
    out = [
        f'<path d="{"".join(f"M{p[0]:.2f},{p[1]:.2f}L{q[0]:.2f},{q[1]:.2f}" for p, q in bonds)}" '
        f'stroke="{ink}" stroke-width="{b * 0.3:.2f}" stroke-linecap="round"/>'
    ]
    out += [
        f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{b * 0.38:.2f}" fill="{ink}"/>'
        for x, y in pts
    ]
    return out


def icon(S, font, ink, top, pot):
    small = S == 32
    w = 40 if small else 16
    mid = (bond(256) + potential(256)) / 2
    body = glyph(font, "s", 256, mid, 220, ink) if small else "\n".join(letters(ink))
    return "\n".join(
        [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{S}" height="{S}" viewBox="0 0 512 512" role="img">',
            LICENSE,
            *([FONT] if small else []),
            "<title>sisl</title>",
            body,
            stroke(curve(potential, X0, X1), pot, w),
            stroke(curve(bond, X0, X1), top, round(w * 1.2, 2)),
            *(
                f'<circle cx="{s:.2f}" cy="{potential(s):.2f}" r="{36 if small else 28}" fill="{pot}"/>'
                for s in SITES
            ),
            "</svg>\n",
        ]
    )


def main(font, outdir="."):
    for name, (ink, top, pot) in COLORS.items():
        for S in (32, 512):
            (Path(outdir) / f"sisl-icon-{name}-{S}.svg").write_text(
                icon(S, font, ink, top, pot)
            )


if __name__ == "__main__":
    main(*sys.argv[1:])
