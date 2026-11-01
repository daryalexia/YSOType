"""Synthetic template grids (BT-Settl or any model atmospheres).

A raw model directory is converted once into a compact ``.npz`` grid sampled
uniformly in ln(lambda); at fit time the grid is convolved to the instrument
resolving power and interpolated onto the observed wavelengths.
"""
import glob
import os
import re
import warnings

import numpy as np
from scipy.ndimage import gaussian_filter1d

_FWHM = 2.0 * np.sqrt(2.0 * np.log(2.0))
# BT-Settl / PHOENIX style: lte030-3.5-0.0a+0.0... or lte03000-3.50-0.0...
_NAME_RE = re.compile(r"lte-?([0-9]+(?:\.[0-9]+)?)-([0-9.]+)")


def parse_model_name(path):
    """Return (Teff, logg) from a BT-Settl/PHOENIX file name."""
    m = _NAME_RE.search(os.path.basename(path))
    if not m:
        raise ValueError("Cannot parse Teff/logg from %s" % path)
    t = float(m.group(1))
    teff = t * 100.0 if t < 1000 else t
    return teff, float(m.group(2))


def read_model_file(path):
    """Read one model spectrum. Returns (wave_um, flux_lambda).

    Supported: SVO/ASCII (lambda[A] flux), BT-Settl ``.7`` files (Fortran 'D'
    exponents, log10 flux with the -8 dex offset) and FITS tables.
    """
    low = path.lower()
    if low.endswith((".fits", ".fits.gz")):
        from astropy.io import fits
        with fits.open(path) as h:
            d = h[1].data
            cols = [c.upper() for c in d.columns.names]
            w = np.asarray(d.field(cols.index("WAVELENGTH") if "WAVELENGTH" in cols else 0), float)
            f = np.asarray(d.field(cols.index("FLUX") if "FLUX" in cols else 1), float)
    else:
        opener = open
        if low.endswith(".gz"):
            import gzip
            opener = gzip.open
        w, f = [], []
        is7 = ".7" in os.path.basename(low) and not low.endswith((".txt", ".dat"))
        with opener(path, "rt") as fh:
            for line in fh:
                if not line.strip() or line.lstrip()[0] in "#!":
                    continue
                parts = line.replace("D", "E").replace("d", "e").split()
                if len(parts) < 2:
                    continue
                try:
                    w.append(float(parts[0]))
                    f.append(float(parts[1]))
                except ValueError:
                    continue
        w, f = np.asarray(w), np.asarray(f)
        if is7:
            f = 10.0 ** (f - 8.0)
    o = np.argsort(w)
    w, f = w[o], f[o]
    keep = np.concatenate([[True], np.diff(w) > 0])
    w, f = w[keep], f[keep]
    wum = w * 1e-4 if np.nanmedian(w) > 100 else w
    return wum, f


def _bin_onto(w_src, f_src, edges):
    """Flux-conserving bin average of a tabulated function onto bins with given edges."""
    cum = np.concatenate([[0.0], np.cumsum(0.5 * (f_src[1:] + f_src[:-1]) * np.diff(w_src))])
    c_edges = np.interp(edges, w_src, cum)
    f_interp = np.interp(0.5 * (edges[1:] + edges[:-1]), w_src, f_src)
    out = np.diff(c_edges) / np.diff(edges)
    # where the source is coarser than the target bins, plain interpolation is better
    dsrc = np.interp(0.5 * (edges[1:] + edges[:-1]), w_src[1:], np.diff(w_src))
    coarse = dsrc > np.diff(edges)
    out[coarse] = f_interp[coarse]
    return out


class TemplateGrid(object):
    """Set of template spectra on a common log-uniform wavelength grid.

    Attributes
    ----------
    wave : (n_w,) micron, uniform in ln(lambda)
    teff, logg : (n_t,)
    flux : (n_t, n_w) F_lambda (arbitrary units)
    resolution : resolving power the grid currently has (np.inf = native)
    """

    def __init__(self, wave, teff, logg, flux, resolution=np.inf, label="BT-Settl"):
        self.wave = np.asarray(wave, float)
        self.teff = np.asarray(teff, float)
        self.logg = np.asarray(logg, float)
        self.flux = np.asarray(flux, float)
        self.resolution = float(resolution)
        self.label = label
        o = np.argsort(-self.teff)  # hot -> cool
        self.teff, self.logg, self.flux = self.teff[o], self.logg[o], self.flux[o]
        self._cache = {}

    def __repr__(self):
        return "<TemplateGrid %s: %d templates, Teff %d-%d K, logg %s, %.2f-%.2f um>" % (
            self.label, len(self.teff), self.teff.min(), self.teff.max(),
            ",".join(sorted(set("%.1f" % g for g in self.logg))), self.wave[0], self.wave[-1])

    # ---- construction ----------------------------------------------------
    @classmethod
    def from_directory(cls, directory, pattern="*", logg=3.5, teff_range=(1200, 5400),
                       wave_range=(0.50, 2.60), sampling=60000.0, label="BT-Settl", verbose=True):
        """Build a grid from a directory of model files.

        ``sampling`` sets the ln-lambda step (dlnlam = 1/sampling); 60000 is
        ample for instruments with R <= 10^4.
        """
        files = sorted(glob.glob(os.path.join(directory, pattern)))
        sel = []
        for f in files:
            try:
                t, g = parse_model_name(f)
            except ValueError:
                continue
            if logg is not None and abs(g - logg) > 1e-3:
                continue
            if teff_range and not (teff_range[0] <= t <= teff_range[1]):
                continue
            sel.append((t, g, f))
        if not sel:
            raise ValueError("No model files with logg=%s and Teff in %s found in %s" % (logg, teff_range, directory))
        dln = 1.0 / sampling
        lnw = np.arange(np.log(wave_range[0]), np.log(wave_range[1]) + dln, dln)
        edges = np.exp(np.concatenate([[lnw[0] - dln / 2], lnw + dln / 2]))
        wave = np.exp(lnw)
        teffs, loggs, fluxes = [], [], []
        for t, g, f in sel:
            w, fl = read_model_file(f)
            if w[0] > wave_range[0] * 1.002 or w[-1] < wave_range[1] * 0.998:
                warnings.warn("%s does not cover %.2f-%.2f um; skipped" % (f, wave_range[0], wave_range[1]))
                continue
            m = (w >= wave_range[0] * 0.99) & (w <= wave_range[1] * 1.01)
            fluxes.append(_bin_onto(w[m], fl[m], edges).astype(np.float32))
            teffs.append(t)
            loggs.append(g)
            if verbose:
                print("  read Teff=%5d logg=%.1f  %s" % (t, g, os.path.basename(f)))
        if not fluxes:
            raise ValueError("None of the selected model files covers %.2f-%.2f um." % wave_range)
        return cls(wave, teffs, loggs, np.array(fluxes), resolution=sampling / 2.0, label=label)

    @classmethod
    def load(cls, path):
        d = np.load(path, allow_pickle=False)
        return cls(d["wave"], d["teff"], d["logg"], d["flux"].astype(float),
                   float(d["resolution"]), str(d["label"]))

    def save(self, path):
        np.savez_compressed(path, wave=self.wave, teff=self.teff, logg=self.logg,
                            flux=self.flux.astype(np.float32), resolution=self.resolution,
                            label=self.label)

    # ---- operations ------------------------------------------------------
    def select(self, teff_min=None, teff_max=None):
        m = np.ones(self.teff.size, bool)
        if teff_min is not None:
            m &= self.teff >= teff_min
        if teff_max is not None:
            m &= self.teff <= teff_max
        return TemplateGrid(self.wave, self.teff[m], self.logg[m], self.flux[m], self.resolution, self.label)

    def convolved(self, R):
        """Grid degraded to resolving power R with a Gaussian LSF (constant in ln lambda)."""
        key = round(float(R), 3)
        if key in self._cache:
            return self._cache[key]
        if not np.isfinite(self.resolution):
            sig_native = 0.0
        else:
            sig_native = 1.0 / (self.resolution * _FWHM)
        sig_target = 1.0 / (R * _FWHM)
        if sig_target <= sig_native:
            warnings.warn("Requested R=%g is not below the grid resolution (%g); no smoothing applied." % (R, self.resolution))
            out = self
        else:
            dln = np.log(self.wave[1] / self.wave[0])
            sig_pix = np.sqrt(sig_target ** 2 - sig_native ** 2) / dln
            flux = gaussian_filter1d(self.flux, sig_pix, axis=1, mode="nearest")
            out = TemplateGrid(self.wave, self.teff, self.logg, flux, R, self.label)
        self._cache[key] = out
        return out

    def on_wavelength(self, wave_um):
        """(n_t, n_pix) template fluxes interpolated at the observed wavelengths."""
        w = np.asarray(wave_um, float)
        if w.min() < self.wave[0] or w.max() > self.wave[-1]:
            warnings.warn("Observed wavelengths extend beyond the template grid; outer pixels are extrapolated flat.")
        lnw = np.log(self.wave)
        x = np.log(np.clip(w, self.wave[0], self.wave[-1]))
        idx = np.clip(np.searchsorted(lnw, x) - 1, 0, lnw.size - 2)
        fr = (x - lnw[idx]) / (lnw[idx + 1] - lnw[idx])
        return self.flux[:, idx] * (1 - fr) + self.flux[:, idx + 1] * fr
