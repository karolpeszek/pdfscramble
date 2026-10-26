#!/usr/local/bin/python3
"""
pdfscramble — shatter every text glyph in a PDF into random Voronoi fragments
while leaving images, plots and diagrams intact. Optionally embeds an invisible
32-bit fingerprint for ownership tracking.

Usage:
    pdfscramble <input.pdf> <output.pdf> [--grid N] [--workers N] [--layers N]
                [--image-ppi N] [--jpeg-quality Q]
                [--fingerprint] [--seed N]
    pdfscramble --extract-fingerprint <pdf>

Pipeline (per page, in parallel):
    pdftocairo -> raw SVG -> flatten_svg() -> scramble_flat_svg()
               -> svg_to_pdf_bytes() -> merge -> embed_fingerprint()
"""
from __future__ import annotations

import argparse
import base64
import io
import math
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from lxml import etree

SVG_NS = "http://www.w3.org/2000/svg"
XLINK_NS = "http://www.w3.org/1999/xlink"
NSMAP = {None: SVG_NS, "xlink": XLINK_NS}

DENSE_PAGE_3 = 2000         # above this many glyphs on a page: 3 pieces per glyph
DENSE_PAGE_2 = 3000         # above this many glyphs on a page: 2 pieces per glyph
LAYERS_MIN, LAYERS_MAX = 4, 16  # default range for the random number of text layers
TARGET_FRAGMENTS = 6000     # per-page fragment budget
USE_DEPTH_LIMIT = 64
LARGE_RECT_PT = 50.0
SEAM_OVERLAP = 0.02         # pt; tiny overlap between opaque fragments hides AA seams

IDENTITY = (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)


# ═════════════════════════════════════════════════════════════════════════════
# Affine math — 6-tuple (a, b, c, d, e, f): x' = a*x + c*y + e, y' = b*x + d*y + f
# ═════════════════════════════════════════════════════════════════════════════

def mat_mul(m1, m2):
    """Return m1 ∘ m2 (apply m2 first, then m1)."""
    a1, b1, c1, d1, e1, f1 = m1
    a2, b2, c2, d2, e2, f2 = m2
    return (a1 * a2 + c1 * b2, b1 * a2 + d1 * b2,
            a1 * c2 + c1 * d2, b1 * c2 + d1 * d2,
            a1 * e2 + c1 * f2 + e1, b1 * e2 + d1 * f2 + f1)


def mat_apply(m, x, y):
    a, b, c, d, e, f = m
    return (a * x + c * y + e, b * x + d * y + f)


def mat_scale_factor(m):
    """Uniform scale approximation used for stroke widths and dash lengths."""
    a, b, c, d, _, _ = m
    return math.sqrt(abs(a * d - b * c)) or 1.0


_NUM_RE = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")
_TRANSFORM_RE = re.compile(r"(matrix|translate|scale|rotate|skewX|skewY)\s*\(([^)]*)\)")


def parse_transform(s):
    m = IDENTITY
    if not s:
        return m
    for kind, args in _TRANSFORM_RE.findall(s):
        v = [float(x) for x in _NUM_RE.findall(args)]
        if kind == "matrix" and len(v) == 6:
            t = tuple(v)
        elif kind == "translate" and v:
            t = (1, 0, 0, 1, v[0], v[1] if len(v) > 1 else 0.0)
        elif kind == "scale" and v:
            t = (v[0], 0, 0, v[1] if len(v) > 1 else v[0], 0, 0)
        elif kind == "rotate" and v:
            r = math.radians(v[0])
            cs, sn = math.cos(r), math.sin(r)
            t = (cs, sn, -sn, cs, 0, 0)
            if len(v) == 3:
                cx, cy = v[1], v[2]
                t = mat_mul(mat_mul((1, 0, 0, 1, cx, cy), t), (1, 0, 0, 1, -cx, -cy))
        elif kind == "skewX" and v:
            t = (1, 0, math.tan(math.radians(v[0])), 1, 0, 0)
        elif kind == "skewY" and v:
            t = (1, math.tan(math.radians(v[0])), 0, 1, 0, 0)
        else:
            continue
        m = mat_mul(m, t)
    return m


def fmt_num(v):
    s = f"{v:.3f}".rstrip("0").rstrip(".")
    return "0" if s in ("-0", "") else s


def fmt_matrix(m):
    return "matrix(" + " ".join(fmt_num(v) if i < 4 else f"{v:.4f}" for i, v in enumerate(m)) + ")"


# ═════════════════════════════════════════════════════════════════════════════
# Path data — parsed into absolute subpaths with only L and C segments.
#   subpath = [start_xy, [("L", p) | ("C", c1, c2, p), ...], closed]
# Quadratics are elevated to cubics and arcs approximated by cubics at parse time,
# so the rest of the pipeline only ever sees lines and cubic beziers.
# ═════════════════════════════════════════════════════════════════════════════

_PATH_TOKEN_RE = re.compile(r"[MmZzLlHhVvCcSsQqTtAa]|[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")


def _arc_to_cubics(x1, y1, rx, ry, phi_deg, large, sweep, x2, y2):
    if (x1, y1) == (x2, y2):
        return []
    rx, ry = abs(rx), abs(ry)
    if rx == 0 or ry == 0:
        return [("L", (x2, y2))]
    phi = math.radians(phi_deg)
    cs, sn = math.cos(phi), math.sin(phi)
    dx, dy = (x1 - x2) / 2, (y1 - y2) / 2
    x1p = cs * dx + sn * dy
    y1p = -sn * dx + cs * dy
    lam = x1p ** 2 / rx ** 2 + y1p ** 2 / ry ** 2
    if lam > 1:
        k = math.sqrt(lam)
        rx, ry = rx * k, ry * k
    num = rx * rx * ry * ry - rx * rx * y1p * y1p - ry * ry * x1p * x1p
    den = rx * rx * y1p * y1p + ry * ry * x1p * x1p
    coef = math.sqrt(max(0.0, num / den)) if den else 0.0
    if large == sweep:
        coef = -coef
    cxp, cyp = coef * rx * y1p / ry, -coef * ry * x1p / rx
    cx = cs * cxp - sn * cyp + (x1 + x2) / 2
    cy = sn * cxp + cs * cyp + (y1 + y2) / 2

    def ang(ux, uy, vx, vy):
        return math.atan2(ux * vy - uy * vx, ux * vx + uy * vy)

    t1 = ang(1, 0, (x1p - cxp) / rx, (y1p - cyp) / ry)
    dt = ang((x1p - cxp) / rx, (y1p - cyp) / ry, (-x1p - cxp) / rx, (-y1p - cyp) / ry)
    if not sweep and dt > 0:
        dt -= 2 * math.pi
    elif sweep and dt < 0:
        dt += 2 * math.pi
    n = max(1, math.ceil(abs(dt) / (math.pi / 2) - 1e-9))
    delta = dt / n
    k = 4 / 3 * math.tan(delta / 4)

    def pt(u, v):
        return (cx + rx * u * cs - ry * v * sn, cy + rx * u * sn + ry * v * cs)

    out = []
    for i in range(n):
        a1 = t1 + i * delta
        a2 = a1 + delta
        c1 = pt(math.cos(a1) - k * math.sin(a1), math.sin(a1) + k * math.cos(a1))
        c2 = pt(math.cos(a2) + k * math.sin(a2), math.sin(a2) - k * math.cos(a2))
        p = (x2, y2) if i == n - 1 else pt(math.cos(a2), math.sin(a2))
        out.append(("C", c1, c2, p))
    return out


def parse_path_d(d):
    toks = _PATH_TOKEN_RE.findall(d or "")
    subpaths = []
    i, n = 0, len(toks)
    cmd = None
    cx = cy = sx = sy = 0.0
    last_ctrl = None       # reflected control for S/T
    last_kind = None
    cur = None

    def num():
        nonlocal i
        v = float(toks[i])
        i += 1
        return v

    def flag():
        nonlocal i
        t = toks[i]
        if len(t) > 1 and t[0] in "01" and t[1] != ".":
            toks[i] = t[1:]       # packed flags, e.g. "a1 1 0 0110 10"
            return t[0] == "1"
        i += 1
        return float(t) != 0

    def ensure_sub():
        nonlocal cur
        if cur is None:
            cur = [(cx, cy), [], False]
            subpaths.append(cur)

    while i < n:
        t = toks[i]
        if t.isalpha():
            cmd = t
            i += 1
            if cmd in "Zz":
                if cur is not None:
                    cur[2] = True
                    cx, cy = sx, sy
                cur = None
                last_kind = "Z"
                continue
        elif cmd is None:
            break
        try:
            rel = cmd.islower()
            C = cmd.upper()
            ox, oy = (cx, cy) if rel else (0.0, 0.0)
            if C == "M":
                x, y = num() + ox, num() + oy
                cx, cy, sx, sy = x, y, x, y
                cur = [(x, y), [], False]
                subpaths.append(cur)
                cmd = "l" if rel else "L"   # implicit lineto after moveto
                last_kind = "M"
                continue
            ensure_sub()
            if C == "L":
                cx, cy = num() + ox, num() + oy
                cur[1].append(("L", (cx, cy)))
            elif C == "H":
                cx = num() + (cx if rel else 0.0)
                cur[1].append(("L", (cx, cy)))
            elif C == "V":
                cy = num() + (cy if rel else 0.0)
                cur[1].append(("L", (cx, cy)))
            elif C == "C":
                c1 = (num() + ox, num() + oy)
                c2 = (num() + ox, num() + oy)
                p = (num() + ox, num() + oy)
                cur[1].append(("C", c1, c2, p))
                last_ctrl, cx, cy = c2, p[0], p[1]
            elif C == "S":
                c1 = (2 * cx - last_ctrl[0], 2 * cy - last_ctrl[1]) if last_kind == "C" else (cx, cy)
                c2 = (num() + ox, num() + oy)
                p = (num() + ox, num() + oy)
                cur[1].append(("C", c1, c2, p))
                last_ctrl, cx, cy = c2, p[0], p[1]
                C = "C"
            elif C in ("Q", "T"):
                if C == "Q":
                    q = (num() + ox, num() + oy)
                else:
                    q = (2 * cx - last_ctrl[0], 2 * cy - last_ctrl[1]) if last_kind == "Q" else (cx, cy)
                p = (num() + ox, num() + oy)
                # elevate quadratic to cubic
                c1 = (cx + 2 / 3 * (q[0] - cx), cy + 2 / 3 * (q[1] - cy))
                c2 = (p[0] + 2 / 3 * (q[0] - p[0]), p[1] + 2 / 3 * (q[1] - p[1]))
                cur[1].append(("C", c1, c2, p))
                last_ctrl, cx, cy = q, p[0], p[1]
                C = "Q"
            elif C == "A":
                rx, ry, phi = num(), num(), num()
                large, sweep = flag(), flag()
                x, y = num() + ox, num() + oy
                cur[1].extend(_arc_to_cubics(cx, cy, rx, ry, phi, large, sweep, x, y))
                cx, cy = x, y
            else:
                i += 1
                continue
            last_kind = C
        except (IndexError, ValueError, TypeError):
            break
    return subpaths


def transform_subpaths(subpaths, m):
    if m == IDENTITY:
        return subpaths
    out = []
    for start, segs, closed in subpaths:
        ns = []
        for s in segs:
            if s[0] == "L":
                ns.append(("L", mat_apply(m, *s[1])))
            else:
                ns.append(("C", mat_apply(m, *s[1]), mat_apply(m, *s[2]), mat_apply(m, *s[3])))
        out.append([mat_apply(m, *start), ns, closed])
    return out


def subpaths_to_d(subpaths):
    parts = []
    for start, segs, closed in subpaths:
        parts.append(f"M{fmt_num(start[0])} {fmt_num(start[1])}")
        for s in segs:
            if s[0] == "L":
                parts.append(f"L{fmt_num(s[1][0])} {fmt_num(s[1][1])}")
            else:
                parts.append("C" + " ".join(f"{fmt_num(p[0])} {fmt_num(p[1])}" for p in s[1:]))
        if closed:
            parts.append("Z")
    return "".join(parts)


def subpaths_bbox(subpaths):
    xs, ys = [], []
    for start, segs, _ in subpaths:
        xs.append(start[0]); ys.append(start[1])
        for s in segs:
            for p in s[1:]:
                xs.append(p[0]); ys.append(p[1])
    if not xs:
        return None
    return min(xs), min(ys), max(xs), max(ys)


def subpaths_have_curves(subpaths):
    return any(s[0] == "C" for _, segs, _ in subpaths for s in segs)


# ═════════════════════════════════════════════════════════════════════════════
# Styles / colours
# ═════════════════════════════════════════════════════════════════════════════

INHERITED_PROPS = (
    "fill", "fill-opacity", "fill-rule", "stroke", "stroke-width", "stroke-opacity",
    "stroke-dasharray", "stroke-dashoffset", "stroke-linecap", "stroke-linejoin",
    "stroke-miterlimit", "clip-rule", "visibility", "color",
)
DEFAULT_STYLE = {
    "fill": "#000000", "fill-opacity": "1", "fill-rule": "nonzero", "stroke": "none",
    "stroke-width": "1", "stroke-opacity": "1", "stroke-dasharray": "none",
    "stroke-dashoffset": "0", "stroke-linecap": "butt", "stroke-linejoin": "miter",
    "stroke-miterlimit": "4", "clip-rule": "nonzero", "visibility": "visible",
    "color": "#000000", "opacity": "1",
}
NAMED_COLORS = {
    "black": "#000000", "white": "#ffffff", "red": "#ff0000", "green": "#008000",
    "blue": "#0000ff", "yellow": "#ffff00", "cyan": "#00ffff", "magenta": "#ff00ff",
    "gray": "#808080", "grey": "#808080", "silver": "#c0c0c0", "maroon": "#800000",
    "navy": "#000080", "olive": "#808000", "purple": "#800080", "teal": "#008080",
    "orange": "#ffa500", "lime": "#00ff00", "transparent": "none",
}


def parse_style_attr(s):
    out = {}
    for decl in (s or "").split(";"):
        if ":" in decl:
            k, v = decl.split(":", 1)
            out[k.strip()] = v.strip()
    return out


def style_to_attr(d):
    return ";".join(f"{k}:{v}" for k, v in d.items())


def parse_color(v):
    """Return '#rrggbb', 'none', or None if unparseable."""
    if v is None:
        return None
    v = v.strip().lower()
    if v in ("none", ""):
        return "none"
    if v in NAMED_COLORS:
        return NAMED_COLORS[v]
    if v.startswith("#"):
        h = v[1:]
        if len(h) in (3, 4):
            h = "".join(ch * 2 for ch in h[:3])
        if len(h) >= 6 and all(ch in "0123456789abcdef" for ch in h[:6]):
            return "#" + h[:6]
        return None
    m = re.match(r"rgba?\(([^)]*)\)", v)
    if m:
        parts = [p.strip() for p in re.split(r"[,\s/]+", m.group(1).strip()) if p.strip()]
        vals = []
        for p in parts[:3]:
            x = float(p[:-1]) * 2.55 if p.endswith("%") else float(p)
            vals.append(max(0, min(255, round(x))))
        if len(vals) == 3:
            return "#%02x%02x%02x" % tuple(vals)
    return None


def fnum(v, default=0.0):
    try:
        m = _NUM_RE.search(str(v))
        return float(m.group(0)) if m else default
    except (TypeError, ValueError):
        return default


def length(v, default=0.0):
    """Parse an SVG length (unit-less or pt/px; % is not supported)."""
    return fnum(v, default) if v is not None else default


def href_of(el):
    return el.get(f"{{{XLINK_NS}}}href") or el.get("href")


def local(tag):
    return tag.rsplit("}", 1)[-1] if isinstance(tag, str) else ""


# ═════════════════════════════════════════════════════════════════════════════
# Step 2 — flatten_svg()
# ═════════════════════════════════════════════════════════════════════════════

SKIP_TAGS = {
    "text", "tspan", "textPath", "animate", "animateTransform", "animateMotion", "set",
    "filter", "mask", "linearGradient", "radialGradient", "pattern", "script", "style",
    "defs", "symbol", "clipPath", "title", "desc", "metadata", "marker", "font", "font-face",
}
CONTAINER_TAGS = {"g", "svg", "a", "switch"}
SHAPE_TAGS = {"path", "rect", "circle", "ellipse", "line", "polyline", "polygon"}


def shape_to_d(el):
    tag = local(el.tag)
    g = el.get
    if tag == "path":
        return g("d") or ""
    if tag == "rect":
        x, y = length(g("x")), length(g("y"))
        w, h = length(g("width")), length(g("height"))
        if w <= 0 or h <= 0:
            return ""
        rx, ry = g("rx"), g("ry")
        rx = length(rx) if rx is not None else None
        ry = length(ry) if ry is not None else None
        if rx is None and ry is None:
            rx = ry = 0.0
        elif rx is None:
            rx = ry
        elif ry is None:
            ry = rx
        rx, ry = min(rx, w / 2), min(ry, h / 2)
        if rx <= 0 or ry <= 0:
            return f"M{x} {y}H{x + w}V{y + h}H{x}Z"
        return (f"M{x + rx} {y}H{x + w - rx}A{rx} {ry} 0 0 1 {x + w} {y + ry}"
                f"V{y + h - ry}A{rx} {ry} 0 0 1 {x + w - rx} {y + h}"
                f"H{x + rx}A{rx} {ry} 0 0 1 {x} {y + h - ry}"
                f"V{y + ry}A{rx} {ry} 0 0 1 {x + rx} {y}Z")
    if tag in ("circle", "ellipse"):
        cx, cy = length(g("cx")), length(g("cy"))
        if tag == "circle":
            rx = ry = length(g("r"))
        else:
            rx, ry = length(g("rx")), length(g("ry"))
        if rx <= 0 or ry <= 0:
            return ""
        return (f"M{cx - rx} {cy}A{rx} {ry} 0 1 0 {cx + rx} {cy}"
                f"A{rx} {ry} 0 1 0 {cx - rx} {cy}Z")
    if tag == "line":
        return f"M{length(g('x1'))} {length(g('y1'))}L{length(g('x2'))} {length(g('y2'))}"
    if tag in ("polyline", "polygon"):
        pts = [float(v) for v in _NUM_RE.findall(g("points") or "")]
        if len(pts) < 4:
            return ""
        d = f"M{pts[0]} {pts[1]}" + "".join(f"L{pts[k]} {pts[k + 1]}" for k in range(2, len(pts) - 1, 2))
        return d + ("Z" if tag == "polygon" else "")
    return ""


class _Flattener:
    def __init__(self, root):
        self.root = root
        self.ids = {}
        for el in root.iter():
            i = el.get("id")
            if i is not None:
                self.ids[i] = el
        self.clip_defs = []          # (new_id, d, clip_rule)
        self.clip_cache = {}

    # -- styles --------------------------------------------------------------
    def compute_style(self, el, parent):
        st = {k: parent[k] for k in INHERITED_PROPS}
        st["opacity"] = parent["opacity"]
        decl = {}
        for k in INHERITED_PROPS + ("opacity",):
            v = el.get(k)
            if v is not None:
                decl[k] = v
        decl.update({k: v for k, v in parse_style_attr(el.get("style")).items()
                     if k in INHERITED_PROPS or k in ("opacity", "display")})
        for k, v in decl.items():
            if k == "display":
                continue
            if v == "inherit":
                continue                                  # keep parent's value
            if k == "opacity":
                st["opacity"] = str(fnum(parent["opacity"], 1) * fnum(v, 1))
            else:
                st[k] = v
        st["_display_none"] = decl.get("display", "").strip() == "none"
        return st

    def resolve_paint(self, v, st):
        if v is None:
            return "none"
        v = v.strip()
        if v == "currentColor":
            v = st.get("color", "#000000")
        if v.startswith("url("):
            m = re.match(r"url\(\s*['\"]?#([^)'\"]+)['\"]?\s*\)\s*(.*)", v)
            if m:
                ref = self.ids.get(m.group(1))
                c = self._paint_server_color(ref) if ref is not None else None
                if c:
                    return c
                fb = parse_color(m.group(2)) if m.group(2) else None
                return fb or "none"
            return "none"
        return parse_color(v) or "#000000"

    def _paint_server_color(self, ref, depth=0):
        """Approximate a gradient/pattern by a single solid colour."""
        if depth > 8 or ref is None:
            return None
        tag = local(ref.tag)
        if tag in ("linearGradient", "radialGradient"):
            stops = [s for s in ref if local(s.tag) == "stop"]
            if not stops:
                h = href_of(ref)
                return self._paint_server_color(self.ids.get(h[1:]) if h else None, depth + 1)
            rgb = []
            for s in stops:
                c = parse_color(parse_style_attr(s.get("style")).get("stop-color") or s.get("stop-color") or "#000")
                if c and c != "none":
                    rgb.append([int(c[k:k + 2], 16) for k in (1, 3, 5)])
            if rgb:
                return "#%02x%02x%02x" % tuple(int(round(v)) for v in np.mean(rgb, axis=0))
        if tag == "pattern":
            for el in ref.iter():
                c = parse_color(el.get("fill") or parse_style_attr(el.get("style")).get("fill"))
                if c and c != "none":
                    return c
        return None

    # -- clip paths ----------------------------------------------------------
    def clip_id_for(self, ref, ctm):
        m = re.match(r"url\(\s*['\"]?#([^)'\"]+)['\"]?\s*\)", ref or "")
        if not m:
            return None
        cp = self.ids.get(m.group(1))
        if cp is None or local(cp.tag) != "clipPath":
            return None
        key = (m.group(1), tuple(round(v, 6) for v in ctm))
        if key in self.clip_cache:
            return self.clip_cache[key]
        base = mat_mul(ctm, parse_transform(cp.get("transform")))
        subs, rules = [], []
        self._collect_clip(cp, base, subs, rules, 0)
        if not subs:
            new_id = None
        else:
            new_id = f"clip{len(self.clip_defs)}"
            rule = rules[0] if len(set(rules)) == 1 and len(rules) == 1 else "nonzero"
            self.clip_defs.append((new_id, subpaths_to_d(subs), rule))
        self.clip_cache[key] = new_id
        return new_id

    def _collect_clip(self, el, ctm, subs, rules, depth):
        if depth > USE_DEPTH_LIMIT:
            return
        for ch in el:
            tag = local(ch.tag)
            m = mat_mul(ctm, parse_transform(ch.get("transform")))
            if tag in SHAPE_TAGS:
                sp = transform_subpaths(parse_path_d(shape_to_d(ch)), m)
                subs.extend(sp)
                rules.append(ch.get("clip-rule") or parse_style_attr(ch.get("style")).get("clip-rule") or "nonzero")
            elif tag == "use":
                ref = self.ids.get((href_of(ch) or "#")[1:])
                if ref is not None:
                    m = mat_mul(m, (1, 0, 0, 1, length(ch.get("x")), length(ch.get("y"))))
                    self._collect_clip_ref(ref, m, subs, rules, depth + 1)
            elif tag in CONTAINER_TAGS:
                self._collect_clip(ch, m, subs, rules, depth + 1)

    def _collect_clip_ref(self, ref, ctm, subs, rules, depth):
        tag = local(ref.tag)
        m = mat_mul(ctm, parse_transform(ref.get("transform")))
        if tag in SHAPE_TAGS:
            subs.extend(transform_subpaths(parse_path_d(shape_to_d(ref)), m))
            rules.append(ref.get("clip-rule") or "nonzero")
        else:
            self._collect_clip(ref, m, subs, rules, depth)

    # -- main walk -----------------------------------------------------------
    def walk(self, el, ctm, parent_style, out, depth):
        if depth > USE_DEPTH_LIMIT:
            return
        tag = local(el.tag)
        if tag in SKIP_TAGS or not isinstance(el.tag, str):
            return
        st = self.compute_style(el, parent_style)
        if st["_display_none"]:
            return
        m = mat_mul(ctm, parse_transform(el.get("transform")))

        # clip-path on this element applies in its own user space (incl. transform)
        target = out
        clip_ref = el.get("clip-path") or parse_style_attr(el.get("style")).get("clip-path")
        if clip_ref and clip_ref != "none":
            cid = self.clip_id_for(clip_ref, m)
            if cid:
                grp = ("clip", cid, [])
                out.append(grp)
                target = grp[2]

        if tag in CONTAINER_TAGS:
            if tag == "svg" and depth > 0:
                m = mat_mul(m, (1, 0, 0, 1, length(el.get("x")), length(el.get("y"))))
            for ch in el:
                self.walk(ch, m, st, target, depth + 1)
        elif tag == "use":
            ref_id = (href_of(el) or "#")[1:]
            ref = self.ids.get(ref_id)
            if ref is None:
                return
            m = mat_mul(m, (1, 0, 0, 1, length(el.get("x")), length(el.get("y"))))
            rtag = local(ref.tag)
            if rtag == "symbol":
                for ch in ref:
                    self.walk(ch, m, st, target, depth + 1)
            else:
                # walk the referenced element, even if it lives inside <defs>
                self.walk_ref(ref, m, st, target, depth + 1)
        elif tag in SHAPE_TAGS:
            if st["visibility"] in ("hidden", "collapse"):
                return
            subs = transform_subpaths(parse_path_d(shape_to_d(el)), m)
            if not subs:
                return
            target.append(("path", subs, self._flat_style(st, m)))
        elif tag == "image":
            if st["visibility"] in ("hidden", "collapse"):
                return
            href = href_of(el) or ""
            if not href.startswith("data:"):
                return
            target.append(("image", {
                "x": length(el.get("x")), "y": length(el.get("y")),
                "width": length(el.get("width")), "height": length(el.get("height")),
                "href": href, "matrix": m, "opacity": fnum(st["opacity"], 1),
            }))

    def walk_ref(self, ref, ctm, parent_style, out, depth):
        tag = local(ref.tag)
        if tag in ("defs", "symbol", "clipPath"):
            return
        if tag in ("g",) or tag in SHAPE_TAGS or tag in ("image", "use", "svg", "a", "switch"):
            self.walk(ref, ctm, parent_style, out, depth)

    def _flat_style(self, st, m):
        k = mat_scale_factor(m)
        fill = self.resolve_paint(st["fill"], st)
        stroke = self.resolve_paint(st["stroke"], st)
        out = {
            "fill": fill,
            "fill-opacity": fmt_num(fnum(st["fill-opacity"], 1)),
            "fill-rule": st["fill-rule"] if st["fill-rule"] in ("evenodd", "nonzero") else "nonzero",
            "stroke": stroke,
            "opacity": fmt_num(fnum(st["opacity"], 1)),
        }
        if stroke != "none":
            out["stroke-width"] = fmt_num(fnum(st["stroke-width"], 1) * k)
            out["stroke-opacity"] = fmt_num(fnum(st["stroke-opacity"], 1))
            out["stroke-linecap"] = st["stroke-linecap"]
            out["stroke-linejoin"] = st["stroke-linejoin"]
            out["stroke-miterlimit"] = fmt_num(fnum(st["stroke-miterlimit"], 4))
            da = st["stroke-dasharray"]
            if da and da.strip() != "none":
                vals = [float(v) * k for v in _NUM_RE.findall(da)]
                if vals and any(v > 0 for v in vals):
                    out["stroke-dasharray"] = " ".join(fmt_num(v) for v in vals)
                    out["stroke-dashoffset"] = fmt_num(fnum(st["stroke-dashoffset"], 0) * k)
        return out


def _items_to_xml(items, parent):
    for it in items:
        kind = it[0]
        if kind == "path":
            attrs = {"d": it[1] if isinstance(it[1], str) else subpaths_to_d(it[1]),
                     "style": style_to_attr(it[2])}
            if len(it) > 3 and it[3]:
                attrs["class"] = it[3]
            etree.SubElement(parent, f"{{{SVG_NS}}}path", attrs)
        elif kind == "image":
            a = it[1]
            el = etree.SubElement(parent, f"{{{SVG_NS}}}image", {
                "x": fmt_num(a["x"]), "y": fmt_num(a["y"]),
                "width": fmt_num(a["width"]), "height": fmt_num(a["height"]),
                "transform": fmt_matrix(a["matrix"]),
                "preserveAspectRatio": "none",
                "style": f"opacity:{fmt_num(a['opacity'])}",
            })
            el.set(f"{{{XLINK_NS}}}href", a["href"])
        elif kind == "clip":
            g = etree.SubElement(parent, f"{{{SVG_NS}}}g", {"clip-path": f"url(#{it[1]})"})
            _items_to_xml(it[2], g)


def _prune_empty(items):
    out = []
    for it in items:
        if it[0] == "clip":
            kids = _prune_empty(it[2])
            if kids:
                out.append(("clip", it[1], kids))
        else:
            out.append(it)
    return out


def svg_page_size(root):
    """Return (width_pt, height_pt, (vb_x, vb_y, vb_w, vb_h))."""
    vb = root.get("viewBox")
    w_attr, h_attr = root.get("width"), root.get("height")

    def to_pt(v):
        if v is None:
            return None
        n = fnum(v, 0)
        if v.endswith("px"):
            return n * 0.75
        if v.endswith("in"):
            return n * 72
        if v.endswith("mm"):
            return n * 72 / 25.4
        if v.endswith("cm"):
            return n * 72 / 2.54
        return n

    w, h = to_pt(w_attr), to_pt(h_attr)
    if vb:
        vals = [float(v) for v in _NUM_RE.findall(vb)]
        vbt = tuple(vals[:4]) if len(vals) >= 4 else (0, 0, w or 612, h or 792)
    else:
        vbt = (0.0, 0.0, w or 612.0, h or 792.0)
    return (w or vbt[2], h or vbt[3], vbt)


def pdftocairo_offset(w_pt, h_pt):
    """Placement error of `pdftocairo -svg -noshrink` for a w×h page.

    pdftocairo treats the output as "paper" and centres the page on a sheet
    rounded up to whole points, so content lands (ceil(w)-w)/2 left and
    (ceil(h)-h)/2 up of where it belongs (verified for odd sizes, rotated
    pages and offset boxes). Without -noshrink it additionally scales the
    whole page by w/ceil(w) — up to ~0.7pt of drift on A4, most visible as
    misaligned grid lines and markers in embedded plots."""
    return ((math.ceil(w_pt - 1e-4) - w_pt) / 2, (math.ceil(h_pt - 1e-4) - h_pt) / 2)


def flatten_svg(svg_bytes, origin_fix=False):
    """Convert pdftocairo SVG to a flat SVG string: only <path>, <image> and
    clip <g> wrappers; every <use> inlined and every transform baked.

    origin_fix=True compensates the page-centring offset of
    `pdftocairo -svg -noshrink` (see pdftocairo_offset)."""
    parser = etree.XMLParser(huge_tree=True, remove_comments=True, resolve_entities=False)
    root = etree.fromstring(svg_bytes, parser)
    w_pt, h_pt, vb = svg_page_size(root)
    base_ctm = IDENTITY
    if origin_fix:
        dx, dy = pdftocairo_offset(w_pt, h_pt)
        base_ctm = (1.0, 0.0, 0.0, 1.0, dx * vb[2] / w_pt, dy * vb[3] / h_pt)
    fl = _Flattener(root)
    items = []
    base = dict(DEFAULT_STYLE)
    root_style = fl.compute_style(root, base)
    for ch in root:
        fl.walk(ch, base_ctm, root_style, items, 1)
    items = _prune_empty(items)

    out = etree.Element(f"{{{SVG_NS}}}svg", nsmap=NSMAP, attrib={
        "width": f"{fmt_num(w_pt)}pt", "height": f"{fmt_num(h_pt)}pt",
        "viewBox": " ".join(fmt_num(v) for v in vb),
    })
    defs = etree.SubElement(out, f"{{{SVG_NS}}}defs")
    for cid, d, rule in fl.clip_defs:
        cp = etree.SubElement(defs, f"{{{SVG_NS}}}clipPath", {"id": cid})
        etree.SubElement(cp, f"{{{SVG_NS}}}path", {"d": d, "style": f"clip-rule:{rule}"})
    _items_to_xml(items, out)
    return etree.tostring(out, encoding="unicode")


# ═════════════════════════════════════════════════════════════════════════════
# Flat SVG reader (shared by scrambler and renderer)
# ═════════════════════════════════════════════════════════════════════════════

def read_flat_svg(svg_text):
    """Return (root, clip_defs{id: (d, rule)}, items) where items mirror the
    flatten representation but keep d strings unparsed."""
    root = etree.fromstring(svg_text.encode("utf-8"), etree.XMLParser(huge_tree=True))
    clips = {}
    for cp in root.iter(f"{{{SVG_NS}}}clipPath"):
        p = cp.find(f"{{{SVG_NS}}}path")
        if p is not None:
            clips[cp.get("id")] = (p.get("d"), parse_style_attr(p.get("style")).get("clip-rule", "nonzero"))

    def read(parent):
        items = []
        for el in parent:
            tag = local(el.tag)
            if tag == "path":
                items.append(("path", el.get("d") or "", parse_style_attr(el.get("style")), el.get("class")))
            elif tag == "image":
                items.append(("image", {
                    "x": fnum(el.get("x")), "y": fnum(el.get("y")),
                    "width": fnum(el.get("width")), "height": fnum(el.get("height")),
                    "href": href_of(el) or "", "matrix": parse_transform(el.get("transform")),
                    "opacity": fnum(parse_style_attr(el.get("style")).get("opacity"), 1),
                }))
            elif tag == "g":
                m = re.match(r"url\(#([^)]+)\)", el.get("clip-path") or "")
                if m:
                    items.append(("clip", m.group(1), read(el)))
                else:
                    items.extend(read(el))
        return items

    return root, clips, read(root)


def write_flat_svg(root_src, clips, items):
    out = etree.Element(f"{{{SVG_NS}}}svg", nsmap=NSMAP, attrib={
        k: root_src.get(k) for k in ("width", "height", "viewBox") if root_src.get(k)})
    defs = etree.SubElement(out, f"{{{SVG_NS}}}defs")
    for cid, (d, rule) in clips.items():
        cp = etree.SubElement(defs, f"{{{SVG_NS}}}clipPath", {"id": cid})
        etree.SubElement(cp, f"{{{SVG_NS}}}path", {"d": d, "style": f"clip-rule:{rule}"})
    _items_to_xml(items, out)
    return etree.tostring(out, encoding="unicode")


# ═════════════════════════════════════════════════════════════════════════════
# Step 3 — scramble_flat_svg()
# ═════════════════════════════════════════════════════════════════════════════

def _should_shatter(style, subpaths):
    fill = style.get("fill") or "none"
    if fill == "none":
        return False
    da = style.get("stroke-dasharray")
    if da and da.strip() != "none":
        return False
    if not subpaths:
        return False
    if not subpaths_have_curves(subpaths):
        bb = subpaths_bbox(subpaths)
        if bb and (bb[2] - bb[0]) > LARGE_RECT_PT and (bb[3] - bb[1]) > LARGE_RECT_PT:
            return False
    return True


def _sample_subpath(start, segs):
    pts = [start]
    cur = start
    for s in segs:
        if s[0] == "L":
            pts.append(s[1])
            cur = s[1]
        else:
            c1, c2, p = s[1], s[2], s[3]
            chord = math.dist(cur, c1) + math.dist(c1, c2) + math.dist(c2, p)
            k = int(min(16, max(4, chord / 0.75)))
            t = np.linspace(0, 1, k + 1)[1:]
            mt = 1 - t
            xs = mt ** 3 * cur[0] + 3 * mt ** 2 * t * c1[0] + 3 * mt * t ** 2 * c2[0] + t ** 3 * p[0]
            ys = mt ** 3 * cur[1] + 3 * mt ** 2 * t * c1[1] + 3 * mt * t ** 2 * c2[1] + t ** 3 * p[1]
            pts.extend(zip(xs.tolist(), ys.tolist()))
            cur = p
    return pts


def build_glyph_shapes(subpaths):
    """Turn a (possibly multi-subpath) glyph into a list of Shapely polygons
    with holes, using containment depth parity rather than difference().

    Outer contours (even depth) become shells; contours directly inside a
    shell (odd depth) become its holes; islands inside holes (e.g. the R
    inside ®) are shells again. Non-contained contours (i-dots, accents)
    come back as independent polygons."""
    from shapely.geometry import Polygon
    from shapely.validation import make_valid

    rings = []
    for start, segs, _ in subpaths:
        pts = _sample_subpath(start, segs)
        if len(pts) < 3:
            continue
        poly = Polygon(pts)
        if not poly.is_valid:
            poly = make_valid(poly)
            parts = [g for g in getattr(poly, "geoms", [poly]) if g.geom_type == "Polygon"]
            if not parts:
                continue
            poly = max(parts, key=lambda g: g.area)
            poly = Polygon(poly.exterior.coords)
        if poly.area < 1e-6:
            continue
        rings.append(poly)
    if not rings:
        return []
    rings.sort(key=lambda p: p.area, reverse=True)

    parent = [-1] * len(rings)
    depth = [0] * len(rings)
    for i, cand in enumerate(rings):
        rp = cand.representative_point()
        # the smallest larger ring that contains this one is its parent
        for j in range(i - 1, -1, -1):
            outer = rings[j]
            if outer.area <= cand.area:
                continue
            if outer.contains(rp) and outer.intersection(cand).area >= 0.9 * cand.area:
                parent[i] = j
                depth[i] = depth[j] + 1
                break

    shells = {}
    for i in range(len(rings)):
        if depth[i] % 2 == 0:
            shells[i] = []
    for i in range(len(rings)):
        if depth[i] % 2 == 1 and parent[i] in shells:
            shells[parent[i]].append(rings[i])

    out = []
    for i, holes in shells.items():
        poly = Polygon(rings[i].exterior.coords, [h.exterior.coords for h in holes])
        if not poly.is_valid:
            poly = make_valid(poly)
        for g in getattr(poly, "geoms", [poly]):
            if g.geom_type == "Polygon" and g.area > 1e-6:
                out.append(g)
    return out


def _poly_to_d(poly):
    """Emit exterior and all interior rings as subpaths of one d string."""
    parts = []
    for ring in [poly.exterior, *poly.interiors]:
        coords = list(ring.coords)
        if len(coords) < 4:
            continue
        parts.append("M" + "L".join(f"{fmt_num(x)} {fmt_num(y)}" for x, y in coords[:-1]) + "Z")
    return "".join(parts)


def _geom_polys(g):
    if g.is_empty:
        return []
    if g.geom_type == "Polygon":
        return [g]
    if hasattr(g, "geoms"):
        return [p for sub in g.geoms for p in _geom_polys(sub)]
    return []


def shatter_polygon_shape(poly, n_seeds, rng_seed, overlap=0.0):
    """Split a Shapely polygon into Voronoi fragments; returns SVG d strings."""
    from scipy.spatial import Voronoi, QhullError
    from shapely.geometry import Polygon
    import shapely

    if n_seeds < 2:
        return [_poly_to_d(poly)]
    rng = np.random.default_rng(rng_seed)
    minx, miny, maxx, maxy = poly.bounds
    w, h = maxx - minx, maxy - miny
    if w <= 0 or h <= 0:
        return [_poly_to_d(poly)]

    # random seeds inside the bbox, preferring ones that land inside the shape
    cand = rng.uniform((minx, miny), (maxx, maxy), size=(n_seeds * 6, 2))
    inside = cand[shapely.contains_xy(poly, cand[:, 0], cand[:, 1])]
    seeds = inside[:n_seeds] if len(inside) >= n_seeds else np.vstack(
        [inside, cand[~shapely.contains_xy(poly, cand[:, 0], cand[:, 1])][:n_seeds - len(inside)]])

    # 8 mirror points far outside the bbox keep every seed's region bounded
    cx, cy, r = (minx + maxx) / 2, (miny + maxy) / 2, 10 * max(w, h)
    mirrors = np.array([[cx + r * math.cos(a), cy + r * math.sin(a)]
                        for a in np.linspace(0, 2 * math.pi, 8, endpoint=False)])
    try:
        vor = Voronoi(np.vstack([seeds, mirrors]))
    except (QhullError, ValueError):
        return [_poly_to_d(poly)]

    frags = []
    for i in range(len(seeds)):
        region = vor.regions[vor.point_region[i]]
        if not region or -1 in region:
            continue
        cell = Polygon(vor.vertices[region])
        if not cell.is_valid:
            cell = cell.buffer(0)
        if overlap > 0:
            cell = cell.buffer(overlap, join_style=2)
        piece = cell.intersection(poly)
        polys = [p for p in _geom_polys(piece) if p.area > 1e-7]
        if polys:
            frags.append("".join(_poly_to_d(p) for p in polys))
    return frags or [_poly_to_d(poly)]


def scramble_flat_svg(svg_text, base_seeds=9, seed=None, n_layers=0):
    """Shatter eligible paths into Voronoi fragments.

    With n_layers > 0 every fragment is assigned to one of n_layers stacked
    layers (class "txt:<k>"). A glyph's fragments go to distinct layers
    whenever it has no more fragments than there are layers, so no single
    layer contains a whole glyph.

    Returns (svg_text, n_fragments, n_shatterable, method)."""
    root, clips, items = read_flat_svg(svg_text)
    rng = random.Random(seed)

    # parse every path once and count eligible ones
    def prep(its):
        out = []
        for it in its:
            if it[0] == "path":
                subs = parse_path_d(it[1])
                out.append(("path", it[1], it[2], it[3], subs, _should_shatter(it[2], subs)))
            elif it[0] == "clip":
                out.append(("clip", it[1], prep(it[2])))
            else:
                out.append(it)
        return out

    def count(its):
        return sum(count(it[2]) if it[0] == "clip" else (1 if it[0] == "path" and it[5] else 0) for it in its)

    pitems = prep(items)
    n_shatterable = count(pitems)

    if n_shatterable == 0:
        return svg_text, 0, 0, "no text"

    # dense pages are never skipped; they just get fewer pieces per glyph
    if n_shatterable > DENSE_PAGE_2:
        n_seeds = 2
    elif n_shatterable > DENSE_PAGE_3:
        n_seeds = 3
    else:
        n_seeds = max(4, min(base_seeds, TARGET_FRAGMENTS // n_shatterable))
    total = 0

    # Every path keeps its original z-order: grid lines, error bars and
    # markers overlap in plots, so moving strokes around visibly breaks them.
    def shatter(its):
        nonlocal total
        out = []
        run = []            # consecutive fragments: shuffled together

        def flush():
            if run:
                rng.shuffle(run)
                if n_layers:
                    # group by layer (stable, so order within a layer stays
                    # random) to keep layer switches in the stream to a minimum
                    run.sort(key=lambda r: int(r[3].split(":")[1]))
                out.extend(run)
                run.clear()

        for it in its:
            if it[0] == "clip":
                flush()
                out.append(("clip", it[1], shatter(it[2])))
                continue
            if it[0] != "path":
                flush()
                out.append(it)
                continue
            _, d, style, cls, subs, eligible = it
            if not eligible:
                flush()
                out.append(("path", d, style, cls))
                continue
            shapes = build_glyph_shapes(subs)
            if not shapes:
                flush()
                out.append(("path", d, style, cls))
                continue
            opacity = fnum(style.get("fill-opacity"), 1) * fnum(style.get("opacity"), 1)
            overlap = SEAM_OVERLAP if opacity >= 0.999 else 0.0
            fstyle = {
                "fill": style["fill"],
                "fill-opacity": style.get("fill-opacity", "1"),
                "opacity": style.get("opacity", "1"),
                "fill-rule": "evenodd",
            }
            biggest = max(s.area for s in shapes)
            frags = []
            for shp in shapes:
                ns = n_seeds if shp.area >= biggest else max(2, round(n_seeds * shp.area / biggest))
                frags.extend(shatter_polygon_shape(shp, ns, rng.getrandbits(63), overlap))
            if n_layers:
                k = min(len(frags), n_layers)
                layers = rng.sample(range(n_layers), k) + [rng.randrange(n_layers) for _ in range(len(frags) - k)]
                rng.shuffle(layers)
                classes = [f"txt:{lay}" for lay in layers]
            else:
                classes = ["txt"] * len(frags)
            for fd, c in zip(frags, classes):
                run.append(("path", fd, fstyle, c))
            total += len(frags)
            # keep any stroke the shape had (e.g. marker edges), drawn
            # directly over its own fragments as in the original
            if (style.get("stroke") or "none") != "none":
                flush()
                sstyle = dict(style)
                sstyle["fill"] = "none"
                out.append(("path", d, sstyle, cls))
        flush()
        return out

    new_items = shatter(pitems)

    return write_flat_svg(root, clips, new_items), total, n_shatterable, f"shattered ×{n_seeds}"


# ═════════════════════════════════════════════════════════════════════════════
# Step 4 — svg_to_pdf_bytes()
# ═════════════════════════════════════════════════════════════════════════════

def layer_name(k):
    """Resource name of text layer k (an optional content group)."""
    return f"L{k}"


def _item_layer(cls):
    if cls and cls.startswith("txt:"):
        return layer_name(int(cls.split(":", 1)[1]))
    return None


PHOTO_MIN_COLORS = 16384      # lossless images with more colours are treated as photos


def prepare_image(raw, mime, disp_w_pt, disp_h_pt, max_ppi=300, jpeg_quality=85):
    """Return (ImageReader, mask) for an embedded image, compressing photos.

    - JPEGs already at <= max_ppi are embedded byte-for-byte (no re-encode).
    - Photos above max_ppi are downsampled to max_ppi and saved as JPEG.
    - Lossless photo-like images (many colours, no alpha) become JPEG.
    - Everything else (plots, line art, transparency) stays lossless.
    Returns None if the image can't be decoded."""
    from reportlab.lib.utils import ImageReader
    from PIL import Image

    try:
        img = Image.open(io.BytesIO(raw))
        img.load()
    except Exception:
        return None
    is_jpeg = mime == "image/jpeg" or img.format == "JPEG"
    has_alpha = "A" in img.getbands() or "transparency" in img.info

    scale = 1.0
    if max_ppi and disp_w_pt > 0 and disp_h_pt > 0:
        scale = min(1.0, max_ppi * disp_w_pt / 72 / img.width, max_ppi * disp_h_pt / 72 / img.height)
    downsample = scale < 0.95      # don't resample for a few percent

    if is_jpeg and not downsample and img.mode in ("RGB", "L", "CMYK"):
        return ImageReader(io.BytesIO(raw)), None

    photo = is_jpeg
    if not photo and not has_alpha:
        thumb = img.convert("RGB")
        thumb.thumbnail((256, 256))
        colors = thumb.getcolors(256 * 256)       # None: more colours than pixels allow
        photo = colors is None or len(colors) >= PHOTO_MIN_COLORS

    if downsample and (photo or img.width * img.height > 4_000_000):
        size = (max(1, round(img.width * scale)), max(1, round(img.height * scale)))
        img = img.resize(size, Image.LANCZOS)

    if photo and not has_alpha:
        img = img.convert("L" if img.mode in ("L", "1") else "RGB")
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=jpeg_quality, optimize=True,
                 subsampling=0 if jpeg_quality >= 90 else 2)
        return ImageReader(io.BytesIO(buf.getvalue())), None

    if img.mode not in ("RGB", "RGBA", "L"):
        img = img.convert("RGBA" if has_alpha else "RGB")
    return ImageReader(img), ("auto" if img.mode == "RGBA" else None)


def svg_to_pdf_bytes(svg_text, max_ppi=300, jpeg_quality=85):
    """Render a flat SVG to one PDF page. Fragments tagged "txt:<k>" are
    wrapped in /OC /L<k> marked content; merge_pages() defines the groups."""
    from reportlab import rl_config
    rl_config.useA85 = 0          # binary streams: ASCII85 adds 25% to everything
    from reportlab.pdfgen import canvas as rl_canvas
    from reportlab.lib.colors import HexColor

    root, clips, items = read_flat_svg(svg_text)
    w_pt, h_pt, (vbx, vby, vbw, vbh) = svg_page_size(root)
    sx, sy = w_pt / (vbw or w_pt), h_pt / (vbh or h_pt)

    buf = io.BytesIO()
    c = rl_canvas.Canvas(buf, pagesize=(w_pt, h_pt), pageCompression=1)
    c.setTitle("")
    c.setAuthor("")
    c.setProducer("")
    c.setCreator("")

    def X(x):
        return (x - vbx) * sx

    def Y(y):
        return h_pt - (y - vby) * sy

    def make_path(subs):
        p = c.beginPath()
        for start, segs, closed in subs:
            p.moveTo(X(start[0]), Y(start[1]))
            for s in segs:
                if s[0] == "L":
                    p.lineTo(X(s[1][0]), Y(s[1][1]))
                else:
                    p.curveTo(X(s[1][0]), Y(s[1][1]), X(s[2][0]), Y(s[2][1]), X(s[3][0]), Y(s[3][1]))
            if closed:
                p.close()
        return p

    layer = [None]

    def set_layer(name):
        if layer[0] == name:
            return
        if layer[0] is not None:
            c._code.append("EMC")
        if name is not None:
            c._code.append(f"/OC /{name} BDC")
        layer[0] = name

    cap = {"butt": 0, "round": 1, "square": 2}
    join = {"miter": 0, "round": 1, "bevel": 2}

    def draw_path(d, st):
        subs = parse_path_d(d)
        if not subs:
            return
        fill = st.get("fill") or "none"
        stroke = st.get("stroke") or "none"
        op = fnum(st.get("opacity"), 1)
        do_fill = fill != "none"
        do_stroke = stroke != "none" and fnum(st.get("stroke-width"), 1) > 0
        if not (do_fill or do_stroke):
            return
        c.saveState()
        if do_fill:
            # colour first: setFillColor resets alpha internally
            c.setFillColor(HexColor(fill))
            c.setFillAlpha(max(0.0, min(1.0, fnum(st.get("fill-opacity"), 1) * op)))
        if do_stroke:
            c.setStrokeColor(HexColor(stroke))
            c.setStrokeAlpha(max(0.0, min(1.0, fnum(st.get("stroke-opacity"), 1) * op)))
            c.setLineWidth(fnum(st.get("stroke-width"), 1) * sx)
            c.setLineCap(cap.get(st.get("stroke-linecap"), 0))
            c.setLineJoin(join.get(st.get("stroke-linejoin"), 0))
            c.setMiterLimit(fnum(st.get("stroke-miterlimit"), 4))
            da = st.get("stroke-dasharray")
            if da and da != "none":
                pat = [float(v) * sx for v in _NUM_RE.findall(da)]
                if len(pat) % 2:
                    pat = pat * 2
                c.setDash(pat, fnum(st.get("stroke-dashoffset"), 0) * sx)
        mode = rl_canvas.FILL_EVEN_ODD if st.get("fill-rule") == "evenodd" else rl_canvas.FILL_NON_ZERO
        c.drawPath(make_path(subs), fill=1 if do_fill else 0, stroke=1 if do_stroke else 0, fillMode=mode)
        c.restoreState()

    def draw_image(a):
        href = a["href"]
        m = re.match(r"data:([^;,]*)(;base64)?,(.*)", href, re.S)
        if not m or a["width"] <= 0 or a["height"] <= 0:
            return
        try:
            raw = base64.b64decode(m.group(3)) if m.group(2) else m.group(3).encode()
        except Exception:
            return
        # page map (SVG viewBox -> PDF) composed with the image's own matrix
        page = (sx, 0, 0, -sy, -vbx * sx, h_pt + vby * sy)
        full = mat_mul(page, a["matrix"])
        disp_w = a["width"] * math.hypot(full[0], full[1])
        disp_h = a["height"] * math.hypot(full[2], full[3])
        prepared = prepare_image(raw, m.group(1), disp_w, disp_h, max_ppi, jpeg_quality)
        if prepared is None:
            return
        reader, mask = prepared
        c.saveState()
        if a["opacity"] < 1:
            c.setFillAlpha(a["opacity"])
        c.transform(*full)
        # image local space is y-down: flip around the image's own box
        c.translate(a["x"], a["y"] + a["height"])
        c.scale(1, -1)
        c.drawImage(reader, 0, 0, a["width"], a["height"], mask=mask)
        c.restoreState()

    def render(its):
        for it in its:
            if it[0] == "path":
                set_layer(_item_layer(it[3]))
                draw_path(it[1], it[2])
            elif it[0] == "image":
                set_layer(None)
                draw_image(it[1])
            elif it[0] == "clip":
                clip = clips.get(it[1])
                if not clip:
                    render(it[2])
                    continue
                set_layer(None)       # keep marked content and q/Q properly nested
                c.saveState()
                mode = rl_canvas.FILL_EVEN_ODD if clip[1] == "evenodd" else rl_canvas.FILL_NON_ZERO
                c.clipPath(make_path(parse_path_d(clip[0])), stroke=0, fill=0, fillMode=mode)
                render(it[2])
                set_layer(None)
                c.restoreState()

    render(items)
    set_layer(None)
    c.showPage()
    c.save()
    return buf.getvalue()


# ═════════════════════════════════════════════════════════════════════════════
# Per-page worker
# ═════════════════════════════════════════════════════════════════════════════

STAGES = ("svg", "flatten", "shatter", "render")


def process_page(input_pdf, page_index, grid_n, tmp_dir, n_layers, seed=None, progress_q=None,
                 max_ppi=300, jpeg_quality=85):
    t0 = time.time()

    def stage(name):
        if progress_q is not None:
            progress_q.put((page_index, name))

    stage("svg")
    svg_path = os.path.join(tmp_dir, f"page_{page_index:05d}.svg")
    subprocess.run(
        # -noshrink: keep 1:1 scale (otherwise the page is shrunk by w/ceil(w))
        ["pdftocairo", "-svg", "-noshrink", "-r", "150", "-f", str(page_index + 1), "-l", str(page_index + 1),
         input_pdf, svg_path],
        check=True, capture_output=True)
    with open(svg_path, "rb") as fh:
        raw = fh.read()
    os.remove(svg_path)

    stage("flatten")
    flat = flatten_svg(raw, origin_fix=True)

    stage("shatter")
    page_seed = None if seed is None else (seed * 1_000_003 + page_index)
    shattered, n_frag, n_glyph, method = scramble_flat_svg(flat, base_seeds=grid_n * grid_n, seed=page_seed,
                                                           n_layers=n_layers)

    stage("render")
    pdf = svg_to_pdf_bytes(shattered, max_ppi=max_ppi, jpeg_quality=jpeg_quality)
    stage("done")
    return page_index, pdf, n_frag, n_glyph, method, time.time() - t0


# ═════════════════════════════════════════════════════════════════════════════
# Merge, optional-content layers, fingerprint
# ═════════════════════════════════════════════════════════════════════════════

def merge_pages(page_pdfs, n_layers, link_source=None):
    """Concatenate single-page PDFs and define the n_layers stacked text
    layers (optional content groups, all visible) that the pages reference.

    link_source: original PDF whose hyperlinks, named destinations and
    bookmarks are carried over (see preserve_links)."""
    from pypdf import PdfReader, PdfWriter
    from pypdf.generic import (ArrayObject, DictionaryObject, NameObject, TextStringObject)

    writer = PdfWriter()
    for data in page_pdfs:
        writer.add_page(PdfReader(io.BytesIO(data)).pages[0])

    if n_layers:
        refs = [writer._add_object(DictionaryObject({
            NameObject("/Type"): NameObject("/OCG"),
            NameObject("/Name"): TextStringObject(f"Layer {k + 1}"),
        })) for k in range(n_layers)]
        for page in writer.pages:
            res = page.get("/Resources")
            res = res.get_object() if res is not None else DictionaryObject()
            props = res.get("/Properties")
            props = props.get_object() if props is not None else DictionaryObject()
            for k, ref in enumerate(refs):
                props[NameObject(f"/{layer_name(k)}")] = ref
            res[NameObject("/Properties")] = props
            page[NameObject("/Resources")] = res
        writer._root_object[NameObject("/OCProperties")] = DictionaryObject({
            NameObject("/OCGs"): ArrayObject(refs),
            NameObject("/D"): DictionaryObject({
                NameObject("/Order"): ArrayObject(refs),
                NameObject("/ON"): ArrayObject(refs),
            }),
        })

    if link_source is not None:
        preserve_links(link_source, writer)

    out = io.BytesIO()
    writer.write(out)
    return out.getvalue()


class _PageMap:
    """Maps a point in an original page's user space to the output page.

    pdftocairo renders the CropBox with /Rotate applied, placed at the
    origin, so output = rotate(point - cropbox origin)."""

    def __init__(self, page):
        box = page.cropbox
        self.x0, self.y0 = float(box.left), float(box.bottom)
        self.w, self.h = float(box.width), float(box.height)
        self.rot = page.rotation % 360

    @property
    def swaps_axes(self):
        return self.rot in (90, 270)

    def point(self, x, y):
        u, v = x - self.x0, y - self.y0
        w, h = self.w, self.h
        return {0: (u, v), 90: (v, w - u), 180: (w - u, h - v), 270: (h - v, u)}.get(self.rot, (u, v))

    def x_of(self, coord, from_y=False):
        """Output x for an original x (or original y when axes swap)."""
        x, y = (0.0, coord) if from_y else (coord, 0.0)
        return self.point(x, y)[0]

    def y_of(self, coord, from_x=False):
        x, y = (coord, 0.0) if from_x else (0.0, coord)
        return self.point(x, y)[1]

    def rect(self, r):
        (ax, ay), (bx, by) = self.point(float(r[0]), float(r[1])), self.point(float(r[2]), float(r[3]))
        return min(ax, bx), min(ay, by), max(ax, bx), max(ay, by)


def _num(v):
    """None for a missing/null PDF number, else float."""
    from pypdf.generic import NullObject
    if v is None or isinstance(v, NullObject):
        return None
    return float(v)


def _map_fit(typ, args, pm):
    """Translate a destination's view (/XYZ, /FitH, ...) to output space."""
    from pypdf.generic import Fit

    args = [_num(a) for a in args]
    sw = pm.swaps_axes

    def ox(left):        # an original x-coordinate → output coordinate
        if left is None:
            return None
        return pm.y_of(left, from_x=True) if sw else pm.x_of(left)

    def oy(top):         # an original y-coordinate → output coordinate
        if top is None:
            return None
        return pm.x_of(top, from_y=True) if sw else pm.y_of(top)

    if typ == "/XYZ":
        left, top, zoom = (args + [None] * 3)[:3]
        if sw:
            return Fit.xyz(oy(top), ox(left), zoom)
        return Fit.xyz(ox(left), oy(top), zoom)
    if typ in ("/FitH", "/FitBH"):
        top = args[0] if args else None
        if sw:
            return (Fit.fit_vertically if typ == "/FitH" else Fit.fit_box_vertically)(oy(top))
        return (Fit.fit_horizontally if typ == "/FitH" else Fit.fit_box_horizontally)(oy(top))
    if typ in ("/FitV", "/FitBV"):
        left = args[0] if args else None
        if sw:
            return (Fit.fit_horizontally if typ == "/FitV" else Fit.fit_box_horizontally)(ox(left))
        return (Fit.fit_vertically if typ == "/FitV" else Fit.fit_box_vertically)(ox(left))
    if typ == "/FitR" and len(args) == 4 and None not in args:
        return Fit.fit_rectangle(*pm.rect(args))
    if typ == "/FitB":
        return Fit.fit_box()
    return Fit.fit()


def _dest_parts(dest):
    """(type, args) of a pypdf Destination."""
    typ = dest.get("/Type", "/Fit")
    keys = {"/XYZ": ("/Left", "/Top", "/Zoom"), "/FitH": ("/Top",), "/FitBH": ("/Top",),
            "/FitV": ("/Left",), "/FitBV": ("/Left",), "/FitR": ("/Left", "/Bottom", "/Right", "/Top")}
    return typ, [dest.get(k) for k in keys.get(typ, ())]


def preserve_links(src_pdf, writer):
    """Carry hyperlinks, named destinations and bookmarks from the original
    PDF over to the rebuilt pages in `writer` (same page order).

    Link annotations are rebuilt from their geometry and action only, so
    nothing else from the original annotation dictionaries is copied. URI,
    GoTo, GoToR and Named actions are kept; JavaScript/Launch are dropped.

    Returns (n_links, n_named_dests, n_bookmarks)."""
    import logging
    from pypdf import PdfReader
    from pypdf.generic import (ArrayObject, Destination, DictionaryObject, FloatObject,
                               NameObject, TextStringObject)

    logging.getLogger("pypdf").setLevel(logging.ERROR)
    reader = PdfReader(src_pdf)
    if len(reader.pages) != len(writer.pages):
        return 0, 0, 0
    maps = [_PageMap(p) for p in reader.pages]
    page_index = {p.indirect_reference.idnum: i for i, p in enumerate(reader.pages)
                  if p.indirect_reference is not None}
    new_refs = [p.indirect_reference for p in writer.pages]

    def explicit_dest(arr):
        """Original explicit destination array → one on the new pages."""
        if not isinstance(arr, (list, ArrayObject)) or not arr:
            return None
        target = arr[0]
        idx = page_index.get(getattr(target, "idnum", None))
        if idx is None and isinstance(target, (int, float)) and not isinstance(target, bool):
            idx = int(target)        # remote-style page number
        if idx is None or not 0 <= idx < len(new_refs):
            return None
        fit = _map_fit(str(arr[1]) if len(arr) > 1 else "/Fit", list(arr[2:]), maps[idx])
        return ArrayObject([new_refs[idx], NameObject(fit.fit_type)] + list(fit.fit_args))

    def dest_value(d):
        d = d.get_object() if hasattr(d, "get_object") else d
        if isinstance(d, NameObject):            # /Dests-dictionary style name
            return TextStringObject(str(d)[1:])
        if isinstance(d, str):                   # name-tree string: kept as is
            return TextStringObject(d)
        return explicit_dest(d)

    # 1) named destinations, re-pointed at the new pages
    n_named = 0
    for name, dest in reader.named_destinations.items():
        idx = reader.get_destination_page_number(dest)
        if idx is None or not 0 <= idx < len(new_refs):
            continue
        typ, args = _dest_parts(dest)
        writer.add_named_destination_object(
            Destination(str(name), new_refs[idx], _map_fit(typ, args, maps[idx])))
        n_named += 1

    # 2) link annotations
    n_links = 0
    for idx, page in enumerate(reader.pages):
        annots = []
        for a in page.get("/Annots") or []:
            a = a.get_object()
            if a.get("/Subtype") != "/Link" or "/Rect" not in a:
                continue
            new = DictionaryObject({
                NameObject("/Type"): NameObject("/Annot"),
                NameObject("/Subtype"): NameObject("/Link"),
                NameObject("/Rect"): ArrayObject([FloatObject(round(v, 4)) for v in maps[idx].rect(a["/Rect"])]),
            })
            for key in ("/Border", "/C", "/H", "/BS", "/F"):
                if key in a:
                    new[NameObject(key)] = a[key]
            if "/QuadPoints" in a:
                q = [float(v) for v in a["/QuadPoints"]]
                pts = [maps[idx].point(q[i], q[i + 1]) for i in range(0, len(q) - 1, 2)]
                new[NameObject("/QuadPoints")] = ArrayObject(
                    [FloatObject(round(c, 4)) for p in pts for c in p])
            if "/Dest" in a:
                d = dest_value(a["/Dest"])
                if d is None:
                    continue
                new[NameObject("/Dest")] = d
            elif "/A" in a:
                act = a["/A"].get_object()
                kind = act.get("/S")
                if kind == "/URI" and "/URI" in act:
                    new_act = {NameObject("/URI"): act["/URI"]}
                elif kind == "/GoTo" and "/D" in act:
                    d = dest_value(act["/D"])
                    if d is None:
                        continue
                    new_act = {NameObject("/D"): d}
                elif kind == "/GoToR" and "/F" in act:
                    new_act = {k: act[k] for k in (NameObject("/F"), NameObject("/D"), NameObject("/NewWindow"))
                               if k in act}
                elif kind == "/Named" and "/N" in act:
                    new_act = {NameObject("/N"): act["/N"]}
                else:
                    continue
                new_act[NameObject("/S")] = kind
                new_act[NameObject("/Type")] = NameObject("/Action")
                new[NameObject("/A")] = DictionaryObject(new_act)
            else:
                continue
            annots.append(writer._add_object(new))
        if annots:
            out_page = writer.pages[idx]
            existing = out_page.get("/Annots")
            arr = ArrayObject(list(existing.get_object())) if existing is not None else ArrayObject()
            arr.extend(annots)
            out_page[NameObject("/Annots")] = arr
            n_links += len(annots)

    # 3) bookmarks (document outline)
    n_marks = 0

    def build(items, parent):
        nonlocal n_marks
        last = None
        for it in items:
            if isinstance(it, list):
                if last is not None:
                    build(it, last)
                continue
            idx = reader.get_destination_page_number(it)
            if idx is None or not 0 <= idx < len(new_refs):
                continue
            typ, args = _dest_parts(it)
            flags = int(it.get("/F", 0) or 0)
            color = it.get("/C")
            last = writer.add_outline_item(
                str(it.title), idx, parent=parent, fit=_map_fit(typ, args, maps[idx]),
                color=tuple(float(c) for c in color) if color is not None and len(color) == 3 else None,
                italic=bool(flags & 1), bold=bool(flags & 2),
                is_open=int(it.get("/Count", 0) or 0) > 0)
            n_marks += 1

    try:
        build(reader.outline, None)
    except Exception:  # a malformed outline must not cost the links
        pass
    mode = reader.trailer["/Root"].get("/PageMode")
    if n_marks and mode is not None:
        writer._root_object[NameObject("/PageMode")] = NameObject(str(mode))
    return n_links, n_named, n_marks


FP_BITS = 32
FP_SLOTS = FP_BITS + 1
FP_MARGIN = 4.0
FP_Y = 4.0
FP_R = 0.5
FP_GS = "GSfp"
_K = 0.5522847498  # circle bezier constant


def _fp_spacing(page_w):
    return (page_w - 2 * FP_MARGIN) / (FP_SLOTS - 1)


def _circle_ops(cx, cy, r):
    k = _K * r
    f = fmt_num
    return (f"{f(cx + r)} {f(cy)} m "
            f"{f(cx + r)} {f(cy + k)} {f(cx + k)} {f(cy + r)} {f(cx)} {f(cy + r)} c "
            f"{f(cx - k)} {f(cy + r)} {f(cx - r)} {f(cy + k)} {f(cx - r)} {f(cy)} c "
            f"{f(cx - r)} {f(cy - k)} {f(cx - k)} {f(cy - r)} {f(cx)} {f(cy - r)} c "
            f"{f(cx + k)} {f(cy - r)} {f(cx + r)} {f(cy - k)} {f(cx + r)} {f(cy)} c h")


def embed_fingerprint(pdf_bytes, fp):
    """Append 33 invisible dots (sync + 32 bits, MSB first) to page 0."""
    from pypdf import PdfReader, PdfWriter
    from pypdf.generic import (ArrayObject, DecodedStreamObject, DictionaryObject,
                               FloatObject, NameObject)

    writer = PdfWriter(clone_from=PdfReader(io.BytesIO(pdf_bytes)))
    page = writer.pages[0]
    box = page.mediabox
    x0, y0 = float(box.left), float(box.bottom)
    page_w = float(box.width)
    spacing = _fp_spacing(page_w)

    dots = []
    for slot in range(FP_SLOTS):
        bit = 1 if slot == 0 else (fp >> (FP_BITS - slot)) & 1
        if bit:
            dots.append(_circle_ops(x0 + FP_MARGIN + slot * spacing, y0 + FP_Y, FP_R))
    ops = f"\nq 0 g /{FP_GS} gs " + " ".join(dots) + " f Q\n"

    res = page.get("/Resources")
    res = res.get_object() if res is not None else DictionaryObject()
    gs = res.get("/ExtGState")
    gs = gs.get_object() if gs is not None else DictionaryObject()
    gs[NameObject(f"/{FP_GS}")] = DictionaryObject({
        NameObject("/Type"): NameObject("/ExtGState"),
        NameObject("/ca"): FloatObject(0),
        NameObject("/CA"): FloatObject(0),
    })
    res[NameObject("/ExtGState")] = gs
    page[NameObject("/Resources")] = res

    stream = DecodedStreamObject()
    stream.set_data(ops.encode("latin-1"))
    stream_ref = writer._add_object(stream)
    contents = page.get("/Contents")
    if contents is None:
        page[NameObject("/Contents")] = stream_ref
    else:
        obj = contents.get_object()
        arr = ArrayObject(list(obj)) if isinstance(obj, ArrayObject) else ArrayObject([contents])
        arr.append(stream_ref)
        page[NameObject("/Contents")] = arr

    out = io.BytesIO()
    writer.write(out)
    return out.getvalue()


def extract_fingerprint(pdf_path):
    """Return the 8-char hex fingerprint, or None if no sync dot is found."""
    from pypdf import PdfReader

    page = PdfReader(pdf_path).pages[0]
    box = page.mediabox
    x0, y0 = float(box.left), float(box.bottom)
    spacing = _fp_spacing(float(box.width))
    contents = page.get_contents()
    data = contents.get_data().decode("latin-1") if contents is not None else ""

    present = set()
    for block in re.findall(rf"/{FP_GS}\s+gs(.*?)\bf\s+Q", data, re.S):
        for xs, ys in re.findall(rf"({_NUM_RE.pattern})\s+({_NUM_RE.pattern})\s+m\b", block):
            x, y = float(xs), float(ys)
            if abs(y - (y0 + FP_Y)) > 0.01:
                continue
            col = (x - FP_R - x0 - FP_MARGIN) / spacing
            slot = round(col)
            if abs(col - slot) < 0.05 and 0 <= slot < FP_SLOTS:
                present.add(slot)
    if 0 not in present:
        return None
    fp = 0
    for slot in range(1, FP_SLOTS):
        fp = (fp << 1) | (1 if slot in present else 0)
    return f"{fp:08x}"


# ═════════════════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════════════════════

def page_count(pdf_path):
    import pypdfium2 as pdfium
    doc = pdfium.PdfDocument(pdf_path)
    try:
        return len(doc)
    finally:
        doc.close()


def run(args):
    from multiprocessing import Manager
    from rich.console import Console, Group
    from rich.live import Live
    from rich.progress import (BarColumn, MofNCompleteColumn, Progress, SpinnerColumn,
                               TextColumn, TimeElapsedColumn)
    from rich.table import Table

    console = Console()
    inp, outp = os.path.abspath(args.input), os.path.abspath(args.output)
    if not os.path.isfile(inp):
        console.print(f"[red]✘ Input not found:[/] {inp}")
        return 1
    if inp == outp:
        console.print("[red]✘ Output must differ from input.[/]")
        return 1
    if shutil.which("pdftocairo") is None:
        console.print("[red]✘ pdftocairo not found.[/] Install poppler "
                      "(macOS: brew install poppler, Linux: apt install poppler-utils).")
        return 1

    n_pages = page_count(inp)
    if n_pages == 0:
        console.print("[red]✘ PDF has no pages.[/]")
        return 1
    n_layers = args.layers
    if n_layers is None:
        n_layers = random.Random(args.seed).randint(LAYERS_MIN, LAYERS_MAX)
    workers = max(1, min(args.workers or os.cpu_count() or 1, n_pages))

    status = {i: {"stage": "queued", "method": "", "frags": None, "time": None} for i in range(n_pages)}
    progress = Progress(SpinnerColumn(), TextColumn("{task.description}"), BarColumn(),
                        MofNCompleteColumn(), TimeElapsedColumn(), console=console)
    overall = progress.add_task("[bold]All pages", total=n_pages)
    page_tasks = {}

    def table():
        t = Table(box=None, pad_edge=False, header_style="bold dim")
        t.add_column("Page", justify="right")
        t.add_column("Status")
        t.add_column("Method")
        t.add_column("Fragments", justify="right")
        t.add_column("Time", justify="right")
        rows = list(status.items())
        if len(rows) > 20:      # keep the live view compact on long documents
            active = [r for r in rows if r[1]["stage"] not in ("queued", "done")]
            done = [r for r in rows if r[1]["stage"] == "done"][-8:]
            queued = [r for r in rows if r[1]["stage"] == "queued"][:4]
            rows = sorted(done + active + queued)
        colors = {"queued": "dim", "done": "green", "error": "red"}
        for i, s in rows:
            t.add_row(str(i + 1), f"[{colors.get(s['stage'], 'yellow')}]{s['stage']}[/]", s["method"],
                      "" if s["frags"] is None else f"{s['frags']:,}",
                      "" if s["time"] is None else f"{s['time']:.1f}s")
        return t

    results = {}
    total_frags = 0
    with tempfile.TemporaryDirectory(prefix="pdfscramble_") as tmp, Manager() as mgr:
        q = mgr.Queue()
        with Live(Group(progress, table()), console=console, refresh_per_second=8) as live, \
                ProcessPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(process_page, inp, i, args.grid, tmp, n_layers, args.seed, q,
                               args.image_ppi, args.jpeg_quality): i
                    for i in range(n_pages)}
            pending = set(futs)
            while pending:
                while not q.empty():
                    idx, st = q.get_nowait()
                    status[idx]["stage"] = st
                    if st in STAGES:
                        if idx not in page_tasks:
                            page_tasks[idx] = progress.add_task(f"Page {idx + 1}", total=len(STAGES))
                        progress.update(page_tasks[idx], completed=STAGES.index(st),
                                        description=f"Page {idx + 1} · {st}")
                done_now = [f for f in pending if f.done()]
                for f in done_now:
                    pending.discard(f)
                    i = futs[f]
                    try:
                        idx, pdf, n_frag, n_glyph, method, dt = f.result()
                    except Exception as e:  # surface which page failed
                        for g in pending:
                            g.cancel()
                        live.stop()
                        console.print(f"[red]✘ Page {i + 1} failed:[/] {type(e).__name__}: {e}")
                        return 1
                    results[idx] = pdf
                    total_frags += n_frag
                    status[idx].update(stage="done", method=method, frags=n_frag, time=dt)
                    if idx in page_tasks:
                        progress.remove_task(page_tasks.pop(idx))
                    progress.advance(overall)
                live.update(Group(progress, table()))
                if pending:
                    time.sleep(0.1)

    try:
        merged = merge_pages([results[i] for i in range(n_pages)], n_layers, link_source=inp)
    except Exception as e:  # never lose the scrambled output over link copying
        console.print(f"[yellow]! Could not preserve hyperlinks ({type(e).__name__}: {e}).[/]")
        merged = merge_pages([results[i] for i in range(n_pages)], n_layers)
    fp_hex = None
    if args.fingerprint:
        fp = random.getrandbits(FP_BITS)
        fp_hex = f"{fp:08x}"
        merged = embed_fingerprint(merged, fp)
    with open(outp, "wb") as fh:
        fh.write(merged)

    console.print(f"\n[bold green]✔ Done[/] → {args.output}")
    console.print(f"  Pages           : {n_pages}")
    console.print(f"  Total fragments : {total_frags:,}")
    console.print(f"  Layers          : {n_layers or 'off'}")
    console.print(f"  Input size      : {os.path.getsize(inp) / 1024:,.1f} KB")
    console.print(f"  Output size     : {os.path.getsize(outp) / 1024:,.1f} KB")
    if fp_hex:
        console.print(f"  Fingerprint     : [bold cyan]{fp_hex}[/]  ← keep this!")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="pdfscramble",
        description="Shatter PDF text glyphs into Voronoi fragments; keep images and plots intact.")
    ap.add_argument("input", nargs="?", help="input PDF")
    ap.add_argument("output", nargs="?", help="output PDF")
    ap.add_argument("--grid", type=int, default=3,
                    help="seed grid size: each glyph gets up to N×N fragments (default 3 → 9)")
    ap.add_argument("--workers", type=int, default=None, help="parallel workers (default: CPU count)")
    ap.add_argument("--layers", type=int, default=None, metavar="N",
                    help=f"spread text fragments across N stacked PDF layers "
                         f"(default: random {LAYERS_MIN}-{LAYERS_MAX}; 0 disables)")
    ap.add_argument("--image-ppi", type=int, default=300, metavar="N",
                    help="downsample photos above N pixels per inch on the page (default 300; 0 = never)")
    ap.add_argument("--jpeg-quality", type=int, default=85, metavar="Q",
                    help="JPEG quality for recompressed photos, 1-95 (default 85)")
    ap.add_argument("--fingerprint", action="store_true", help="embed an invisible 32-bit fingerprint")
    ap.add_argument("--seed", type=int, default=None, help="random seed for reproducible shattering")
    ap.add_argument("--extract-fingerprint", metavar="PDF", help="read the fingerprint from a PDF")
    args = ap.parse_args(argv)

    if args.extract_fingerprint:
        fp = extract_fingerprint(args.extract_fingerprint)
        if fp is None:
            print("No fingerprint found.", file=sys.stderr)
            return 1
        print(fp)
        return 0
    if not args.input or not args.output:
        ap.error("input and output PDFs are required")
    if args.layers is not None and not (args.layers == 0 or 2 <= args.layers <= 64):
        ap.error("--layers must be 0 or between 2 and 64")
    if not 1 <= args.jpeg_quality <= 95:
        ap.error("--jpeg-quality must be between 1 and 95")
    if args.image_ppi < 0:
        ap.error("--image-ppi must be >= 0")
    if args.grid < 1:
        ap.error("--grid must be >= 1")
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
