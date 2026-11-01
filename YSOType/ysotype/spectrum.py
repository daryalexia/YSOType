"""Observed spectra: reading, unit handling, masking and normalisation.

Wavelengths are always stored in micron internally.
"""
import os
import warnings

import numpy as np

# ---------------------------------------------------------------------------
# Default masks (micron)
# ---------------------------------------------------------------------------
TELLURIC_BANDS = [
    (0.6860, 0.6950),   # O2 B band
    (0.7590, 0.7700),   # O2 A band
    (0.8950, 0.9800),   # H2O
    (1.1100, 1.1600),   # H2O
    (1.3300, 1.4900),   # H2O, between J and H
    (1.7800, 1.9800),   # H2O, between H and K
    (2.4500, 2.6000),   # red edge of K
]

# Emission lines common in accreting / outflowing YSOs (vacuum-ish, micron)
EMISSION_LINES = {
    "Hbeta": 0.48613, "HeI5876": 0.58756, "NaD": 0.58924, "[OI]6300": 0.63003,
    "[OI]6363": 0.63637, "Halpha": 0.65628, "[NII]6583": 0.65835, "HeI6678": 0.66782,
    "[SII]6716": 0.67164, "[SII]6731": 0.67308, "HeI7065": 0.70652, "OI8446": 0.84465,
    "CaII8498": 0.84980, "CaII8542": 0.85421, "CaII8662": 0.86621,
    "Pa_delta": 1.00494, "HeI10830": 1.08303, "Pa_gamma": 1.09381,
    "[FeII]12567": 1.25668, "Pa_beta": 1.28181, "[FeII]16440": 1.64400,
    "Br12": 1.64117, "Br11": 1.68111, "Br10": 1.73669,
    "H2_2.12": 2.12183, "Br_gamma": 2.16612, "H2_2.25": 2.24771,
}


def line_halfwidth(lam0, R, v_kms=300.0):
    """Same prescription as the accretion analysis: max(3 lam/R, lam v/c)."""
    return np.maximum(3.0 * lam0 / R, lam0 * v_kms / 2.99792458e5)


def ranges_to_mask(wave, ranges):
    """Boolean array, True inside any (lo, hi) range."""
    m = np.zeros(wave.shape, bool)
    for lo, hi in ranges:
        m |= (wave >= lo) & (wave <= hi)
    return m


def default_bad_regions(R, tellurics=True, lines=True, v_kms=300.0):
    regs = []
    if tellurics:
        regs += list(TELLURIC_BANDS)
    if lines:
        for lam in EMISSION_LINES.values():
            hw = float(line_halfwidth(lam, R, v_kms))
            regs.append((lam - hw, lam + hw))
    return regs


# ---------------------------------------------------------------------------
# Spectrum container
# ---------------------------------------------------------------------------
class Spectrum(object):
    """1-D spectrum with wavelength in micron."""

    def __init__(self, wave, flux, err=None, name="target", instrument="", meta=None):
        wave = np.asarray(wave, float)
        flux = np.asarray(flux, float)
        if err is None:
            err = estimate_noise(flux)
            self.err_estimated = True
        else:
            self.err_estimated = False
        err = np.asarray(err, float)
        order = np.argsort(wave)
        self.wave, self.flux, self.err = wave[order], flux[order], err[order]
        self.name = name
        self.instrument = instrument
        self.meta = dict(meta or {})

    def __repr__(self):
        return "<Spectrum %s %s: %.3f-%.3f um, %d px>" % (
            self.name, self.instrument, self.wave.min(), self.wave.max(), self.wave.size)

    @property
    def good(self):
        return np.isfinite(self.wave) & np.isfinite(self.flux) & np.isfinite(self.err) & (self.err > 0)

    def trimmed(self, wmin=None, wmax=None):
        m = np.ones(self.wave.shape, bool)
        if wmin is not None:
            m &= self.wave >= wmin
        if wmax is not None:
            m &= self.wave <= wmax
        return Spectrum(self.wave[m], self.flux[m], self.err[m], self.name, self.instrument, self.meta)

    def norm_factor(self, lam, width=0.01, exclude=None):
        """Median flux in a window around ``lam`` (micron), skipping ``exclude`` ranges."""
        m = (np.abs(self.wave - lam) <= width / 2.0) & self.good
        if exclude:
            m &= ~ranges_to_mask(self.wave, exclude)
        if m.sum() < 3:
            return np.nan
        return float(np.median(self.flux[m]))

    def pixels_per_resel(self, R):
        dl = np.median(np.diff(self.wave))
        return max(1.0, float(np.median(self.wave) / R / dl))


def estimate_noise(flux, box=21):
    """Rough per-pixel noise from the scatter of first differences (used when no error array)."""
    f = np.asarray(flux, float)
    d = np.abs(np.diff(f))
    d = np.concatenate([d[:1], d])
    from scipy.ndimage import median_filter
    sig = 1.4826 * median_filter(np.nan_to_num(d), size=box, mode="nearest") / np.sqrt(2.0)
    floor = np.nanmedian(sig[sig > 0]) if np.any(sig > 0) else 1.0
    sig[~(sig > 0)] = floor
    return sig


# ---------------------------------------------------------------------------
# Readers
# ---------------------------------------------------------------------------
_WAVE_KEYS = ("WAVE", "WAVELENGTH", "LAMBDA", "LAM", "WAVE_AIR", "WAVE_VAC")
_FLUX_KEYS = ("FLUX", "FLUX_REDUCED", "FLUX_CAL", "SPEC", "F")
_ERR_KEYS = ("ERR", "ERROR", "SIGMA", "FLUX_ERR", "ERR_REDUCED", "UNCERTAINTY", "NOISE")
_IVAR_KEYS = ("IVAR", "INVVAR")


def to_micron(wave, unit=None):
    """Convert to micron. ``unit`` in {'um','nm','A'}; guessed from the median if None."""
    wave = np.asarray(wave, float)
    if unit is None:
        med = np.nanmedian(wave)
        if med < 10:
            unit = "um"
        elif med < 5000:
            unit = "nm"
        else:
            unit = "A"
    unit = unit.lower().replace("angstrom", "a").replace("micron", "um")
    scale = {"um": 1.0, "nm": 1e-3, "a": 1e-4}[unit]
    return wave * scale, unit


def read_spectrum(path, wave_unit=None, name=None, instrument="", ext=None):
    """Read a 1-D spectrum from FITS (image or table) or ASCII (2-3 columns)."""
    name = name or os.path.splitext(os.path.basename(path))[0]
    low = path.lower()
    if low.endswith((".fits", ".fit", ".fits.gz", ".fts")):
        wave, flux, err, meta = _read_fits(path, ext)
    else:
        data = np.genfromtxt(path, comments="#", invalid_raise=False)
        data = np.atleast_2d(data)
        if data.shape[1] < 2:
            raise ValueError("%s: need at least 2 columns (wavelength, flux[, error])" % path)
        wave, flux = data[:, 0], data[:, 1]
        err = data[:, 2] if data.shape[1] > 2 else None
        meta = {}
    wave, unit = to_micron(wave, wave_unit)
    meta["wave_unit_in"] = unit
    sp = Spectrum(wave, flux, err, name=name, instrument=instrument or meta.get("INSTRUME", ""), meta=meta)
    if sp.err_estimated:
        warnings.warn("%s: no error array found; noise estimated from the data." % path)
    return sp


def _read_fits(path, ext=None):
    try:
        from astropy.io import fits
    except ImportError:
        raise ImportError("Reading FITS files requires astropy (pip install astropy).")
    with fits.open(path) as hdul:
        hdus = [hdul[ext]] if ext is not None else list(hdul)
        for h in hdus:
            if h.data is None:
                continue
            if isinstance(h, (fits.BinTableHDU, fits.TableHDU)):
                cols = {c.upper(): c for c in h.columns.names}
                wk = next((cols[k] for k in _WAVE_KEYS if k in cols), None)
                fk = next((cols[k] for k in _FLUX_KEYS if k in cols), None)
                if wk is None or fk is None:
                    continue
                wave = np.ravel(h.data[wk]).astype(float)
                flux = np.ravel(h.data[fk]).astype(float)
                ek = next((cols[k] for k in _ERR_KEYS if k in cols), None)
                ik = next((cols[k] for k in _IVAR_KEYS if k in cols), None)
                if ek is not None:
                    err = np.ravel(h.data[ek]).astype(float)
                elif ik is not None:
                    iv = np.ravel(h.data[ik]).astype(float)
                    err = np.where(iv > 0, 1 / np.sqrt(np.abs(iv)), np.nan)
                else:
                    err = None
                meta = dict((k, hdul[0].header.get(k)) for k in ("OBJECT", "INSTRUME", "DATE-OBS") if k in hdul[0].header)
                return wave, flux, err, meta
            # image HDU with a linear (or log-linear) wavelength solution
            data = np.asarray(h.data, float)
            hdr = h.header
            flux = data if data.ndim == 1 else data.reshape(-1, data.shape[-1])[0]
            n = flux.size
            crval = hdr.get("CRVAL1")
            cdelt = hdr.get("CDELT1", hdr.get("CD1_1"))
            if crval is None or cdelt is None:
                continue
            crpix = hdr.get("CRPIX1", 1.0)
            wave = crval + (np.arange(n) + 1 - crpix) * cdelt
            if hdr.get("DC-FLAG", 0) == 1 or str(hdr.get("CTYPE1", "")).upper().startswith("LOG"):
                wave = 10 ** wave
            err = None
            if data.ndim > 1 and data.reshape(-1, n).shape[0] > 1:
                err = data.reshape(-1, n)[-1]  # last plane is often sigma
            else:
                for hh in hdul:
                    en = str(hh.header.get("EXTNAME", "")).upper()
                    if hh is not h and en in ("ERR", "ERRS", "SIGMA", "ERROR", "VARIANCE", "VAR") and hh.data is not None:
                        e = np.ravel(hh.data).astype(float)[:n]
                        err = np.sqrt(np.abs(e)) if en.startswith("VAR") else e
                        break
            meta = dict((k, hdr.get(k, hdul[0].header.get(k))) for k in ("OBJECT", "INSTRUME", "DATE-OBS"))
            return wave, flux, err, meta
    raise ValueError("%s: no usable spectrum found (table with WAVE/FLUX columns or image with CRVAL1/CDELT1)." % path)
