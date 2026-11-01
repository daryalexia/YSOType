"""Spectral indices and the index-based first guess of the spectral type."""
import os

import numpy as np

from .extinction import transmission

_DATA = os.path.join(os.path.dirname(__file__), "data")


def _ranges(txt):
    out = []
    for part in txt.split(";"):
        lo, hi = part.split("-")
        out.append((float(lo), float(hi)))
    return out


def load_index_definitions(path=None):
    path = path or os.path.join(_DATA, "indices.csv")
    defs = []
    with open(path) as fh:
        header = None
        for line in fh:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            parts = [p.strip() for p in s.split(",")]
            if header is None:
                header = parts
                continue
            row = dict(zip(header, parts))
            row["band"] = _ranges(row["band"])
            row["cont"] = _ranges(row["cont"])
            defs.append(row)
    return defs


def _mean_in(wave, flux, err, ranges):
    vals, errs = [], []
    for lo, hi in ranges:
        m = (wave >= lo) & (wave <= hi) & np.isfinite(flux)
        if m.sum() < 2:
            return np.nan, np.nan
        vals.append(np.mean(flux[m]))
        errs.append(np.sqrt(np.sum(err[m] ** 2)) / m.sum() if err is not None else 0.0)
    return np.mean(vals), np.sqrt(np.sum(np.square(errs))) / len(errs)


def measure_indices(wave, flux, err=None, defs=None):
    """Return {name: (value, sigma)}; NaN where the spectrum does not cover the bands."""
    defs = defs or load_index_definitions()
    out = {}
    for d in defs:
        b, eb = _mean_in(wave, flux, err, d["band"])
        c, ec = _mean_in(wave, flux, err, d["cont"])
        if not (np.isfinite(b) and np.isfinite(c)) or c == 0:
            out[d["name"]] = (np.nan, np.nan)
            continue
        v = b / c
        s = abs(v) * np.sqrt((eb / b) ** 2 + (ec / c) ** 2) if err is not None and b != 0 else np.nan
        out[d["name"]] = (v, s)
    return out


def index_first_guess(wave, flux, err, template_flux, teff, av=0.0, rv=5.0, kinds=("temperature",),
                      defs=None, floor=0.01):
    """Nearest template in index space (templates reddened by ``av``).

    Returns (teff_best, distance_per_template, measured_indices).
    """
    defs = [d for d in (defs or load_index_definitions()) if d["kind"] in kinds]
    obs = measure_indices(wave, flux, err, defs)
    names = [d["name"] for d in defs if np.isfinite(obs[d["name"]][0])]
    if not names:
        return np.nan, None, obs
    ext = transmission(wave, av, rv) if av else 1.0
    dist = np.zeros(len(teff))
    for i in range(len(teff)):
        tm = measure_indices(wave, template_flux[i] * ext, None, [d for d in defs if d["name"] in names])
        for n in names:
            v, s = obs[n]
            sig = max(s if np.isfinite(s) else 0.0, floor * abs(v))
            dist[i] += ((v - tm[n][0]) / sig) ** 2
    dist /= len(names)
    return float(teff[np.argmin(dist)]), dist, obs
