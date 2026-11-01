"""Spectral-type bookkeeping: string <-> numeric code <-> effective temperature.

Numeric code (one unit = one subclass, K7 -> M0 is one step):

    G0..G9   -> -10 .. -1
    K0..K7   ->   0 ..  7
    M0..M9.5 ->   8 .. 17.5
    L0..L9   ->  18 .. 27

The SpT--Teff relation is read from ``data/spt_teff.csv`` so that the adopted
scale (e.g. Pecaut & Mamajek 2013 for <M5, Herczeg & Hillenbrand 2014 for
later M types, Filippazzo et al. 2015 for young L dwarfs) can be replaced
without touching the code.
"""
import os
import re
import warnings

import numpy as np

_DATA = os.path.join(os.path.dirname(__file__), "data")
_CLASS_OFFSET = {"G": -10.0, "K": 0.0, "M": 8.0, "L": 18.0}
_SPT_RE = re.compile(r"^\s*([GKML])\s*([0-9]+(?:\.[0-9]+)?)\s*$", re.I)


def spt_to_code(spt):
    """'M5.5' -> 13.5. Numbers are returned unchanged."""
    if spt is None:
        return None
    if isinstance(spt, (int, float, np.floating, np.integer)):
        return float(spt)
    m = _SPT_RE.match(str(spt))
    if not m:
        raise ValueError("Cannot parse spectral type %r (expected e.g. 'K7', 'M5.5', 'L1')" % spt)
    cls, sub = m.group(1).upper(), float(m.group(2))
    if cls == "K" and sub > 7.75:
        raise ValueError("K subtypes run K0-K7 in this scheme (K7 -> M0 is one step): %r" % spt)
    return _CLASS_OFFSET[cls] + sub


def code_to_spt(code, step=0.25):
    """13.5 -> 'M5.5'. The code is first rounded to the nearest ``step``."""
    if code is None or not np.isfinite(code):
        return "--"
    c = round(float(code) / step) * step
    for cls, lo, hi in (("G", -10, 0), ("K", 0, 8), ("M", 8, 18), ("L", 18, 1e9)):
        if lo <= c < hi or (cls == "G" and c < 0):
            sub = c - _CLASS_OFFSET[cls]
            txt = ("%.2f" % sub).rstrip("0").rstrip(".")
            return "%s%s" % (cls, txt)
    return "--"


class TeffScale(object):
    """Monotonic SpT <-> Teff relation interpolated in (code, Teff)."""

    def __init__(self, path=None):
        path = path or os.path.join(_DATA, "spt_teff.csv")
        self.path = path
        codes, teffs, unverified = [], [], False
        with open(path) as fh:
            for line in fh:
                s = line.strip()
                if not s:
                    continue
                if s.startswith("#"):
                    if "UNVERIFIED" in s.upper():
                        unverified = True
                    continue
                if s.lower().startswith("spt"):
                    continue
                parts = [p.strip() for p in s.split(",")]
                codes.append(spt_to_code(parts[0]))
                teffs.append(float(parts[1]))
        order = np.argsort(codes)
        self.code = np.asarray(codes, float)[order]
        self.teff = np.asarray(teffs, float)[order]
        if np.any(np.diff(self.teff) >= 0):
            raise ValueError("Teff must decrease monotonically with spectral type in %s" % path)
        if unverified:
            warnings.warn(
                "The SpT-Teff table %s is marked UNVERIFIED. Replace it with the exact "
                "values of your adopted scale before publishing results." % path)

    def teff_to_code(self, teff):
        t = np.asarray(teff, float)
        # np.interp needs increasing x
        return np.interp(t, self.teff[::-1], self.code[::-1], left=np.nan, right=np.nan)

    def code_to_teff(self, code):
        return np.interp(np.asarray(code, float), self.code, self.teff, left=np.nan, right=np.nan)

    def teff_to_spt(self, teff, step=0.25):
        return code_to_spt(float(self.teff_to_code(teff)), step=step)

    def spt_to_teff(self, spt):
        return float(self.code_to_teff(spt_to_code(spt)))
