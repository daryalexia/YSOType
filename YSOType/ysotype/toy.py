"""Toy templates and mock spectra for tests and the ``ysotype demo`` command.

These are NOT physical model atmospheres: a blackbody with Teff-dependent
Gaussian 'molecular bands'. They exist only to exercise the code.
"""
import numpy as np

from .extinction import transmission
from .spectrum import Spectrum
from .templates import TemplateGrid

_BANDS = [  # centre um, width um, depth scaling with coolness
    (0.715, 0.010, 0.5), (0.745, 0.008, 0.4), (0.790, 0.010, 0.4), (0.845, 0.008, 0.3),
    (0.819, 0.0015, 0.3), (0.99, 0.004, 0.2), (1.177, 0.0015, 0.25), (1.250, 0.0015, 0.25),
    (1.40, 0.05, 0.6), (1.55, 0.05, 0.3), (1.80, 0.06, 0.5), (2.294, 0.006, 0.3),
    (2.35, 0.03, 0.25), (2.206, 0.0015, 0.2),
]


def _bb(wave_um, teff):
    x = 14387.77 / (wave_um * teff)
    return 1.0 / (wave_um ** 5 * np.expm1(x))


def toy_grid(teffs=None, sampling=20000.0, wave_range=(0.5, 2.6)):
    teffs = np.arange(2000, 4601, 100) if teffs is None else np.asarray(teffs, float)
    dln = 1.0 / sampling
    wave = np.exp(np.arange(np.log(wave_range[0]), np.log(wave_range[1]), dln))
    flux = []
    for t in teffs:
        f = _bb(wave, t)
        cool = np.clip((4800.0 - t) / 2800.0, 0, 1.2)
        prof = np.ones_like(wave)
        for c, wdt, dep in _BANDS:
            prof *= 1 - min(0.95, dep * cool * (1 + 0.5 * np.sin(c * 7))) * np.exp(-0.5 * ((wave - c) / wdt) ** 2)
        flux.append(f * prof)
    flux = np.array(flux)
    flux /= np.median(flux, axis=1, keepdims=True)
    return TemplateGrid(wave, teffs, np.full(teffs.size, 3.5), flux, resolution=sampling / 2.0, label="toy")


def mock_spectrum(grid, teff, av, r, R=3000.0, wave_range=(0.60, 1.00), snr=50.0, rv=5.0,
                  ref=0.75, seed=1, name="mock", pix_per_resel=3.0):
    """Mock observation of a toy template with extinction, flat veiling and noise."""
    rng = np.random.RandomState(seed)
    g = grid.convolved(R)
    mid = 0.5 * (wave_range[0] + wave_range[1])
    dl = mid / R / pix_per_resel
    wave = np.arange(wave_range[0], wave_range[1], dl)
    i = int(np.argmin(np.abs(g.teff - teff)))
    T = g.on_wavelength(wave)[i]
    tref = np.median(g.flux[i, np.abs(g.wave - ref) < 0.003])
    f = (T + r * tref) * transmission(wave, av, rv)
    f /= np.median(f)
    err = f / snr
    return Spectrum(wave, f + rng.normal(0, 1, wave.size) * err, err, name=name, instrument="mock")
