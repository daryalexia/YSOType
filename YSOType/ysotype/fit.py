"""Simultaneous fit of spectral type, extinction and veiling.

Model (per fitted segment j, e.g. optical or J/H/K):

    f_mod(lam) = C * [ T(lam) + E_j ] * 10**(-0.4 A_V a(lam))

with T the template convolved to the instrument resolution, a(lam) =
A_lam/A_V (Fitzpatrick 1999), and E_j a flat excess continuum. The veiling is
reported as r_j = E_j / T(lam_ref,j). For every template and every A_V on the
grid, C and the E_j are *linear* parameters and are solved exactly by
non-negative weighted least squares, so the chi^2 surface over (Teff, A_V) is
exact rather than sampled in r.
"""
import itertools
import warnings

import numpy as np

from .extinction import a_lambda_over_av
from .indices import index_first_guess
from .spectrum import default_bad_regions, ranges_to_mask
from .spt import TeffScale, spt_to_code

DEFAULT_SEGMENTS = [("opt", 0.55, 1.00), ("J", 1.00, 1.35), ("H", 1.45, 1.82), ("K", 1.95, 2.50)]
DEFAULT_REF = {"opt": 0.75, "J": 1.25, "H": 1.65, "K": 2.20}


class FitConfig(object):
    """Fit settings (all wavelengths in micron)."""

    def __init__(self, av_min=0.0, av_max=60.0, av_step=0.1, rv=5.0, veiling="single",
                 segments=None, ref_wave=None, fit_range=None, mask_tellurics=True,
                 mask_lines=True, line_velocity=300.0, extra_masks=None, min_snr=3.0,
                 min_pixels=30, correlated_errors=True, initial_spt=None, initial_av=None,
                 search_window=None, teff_min=None, teff_max=None, scale_path=None):
        if veiling not in ("none", "single", "per-band"):
            raise ValueError("veiling must be 'none', 'single' or 'per-band'")
        self.av_grid = np.round(np.arange(av_min, av_max + av_step / 2.0, av_step), 6)
        self.rv = rv
        self.veiling = veiling
        self.segments = segments or DEFAULT_SEGMENTS
        self.ref_wave = ref_wave
        self.fit_range = fit_range
        self.mask_tellurics = mask_tellurics
        self.mask_lines = mask_lines
        self.line_velocity = line_velocity
        self.extra_masks = list(extra_masks or [])
        self.min_snr = min_snr
        self.min_pixels = min_pixels
        self.correlated_errors = correlated_errors
        self.initial_spt = initial_spt
        self.initial_av = initial_av
        self.search_window = search_window  # (d_subclass, d_av) around the initial guess
        self.teff_min = teff_min
        self.teff_max = teff_max
        self.scale_path = scale_path


class Fitter(object):
    """Prepares a spectrum + template grid and computes the chi^2 cube."""

    def __init__(self, spectrum, grid, resolving_power, config=None):
        self.spec = spectrum
        self.R = float(resolving_power)
        self.cfg = config or FitConfig()
        self.scale = TeffScale(self.cfg.scale_path)
        g = grid.select(self.cfg.teff_min, self.cfg.teff_max)
        if g.teff.size < 3:
            raise ValueError("Fewer than 3 templates within the requested Teff range.")
        self.grid = g.convolved(self.R)
        self.teff = self.grid.teff
        self.T_all = self.grid.on_wavelength(spectrum.wave)          # (n_t, n_pix)
        self.alam = a_lambda_over_av(spectrum.wave, self.cfg.rv)     # (n_pix,)
        self.ppr = spectrum.pixels_per_resel(self.R)
        self.user_masks = list(self.cfg.extra_masks)

    # ---- pixel selection ------------------------------------------------
    def bad_regions(self):
        regs = default_bad_regions(self.R, self.cfg.mask_tellurics, self.cfg.mask_lines, self.cfg.line_velocity)
        return regs + self.user_masks

    def fit_mask(self):
        s = self.spec
        m = s.good.copy()
        if self.cfg.fit_range:
            m &= (s.wave >= self.cfg.fit_range[0]) & (s.wave <= self.cfg.fit_range[1])
        m &= ~ranges_to_mask(s.wave, self.bad_regions())
        if self.cfg.min_snr:
            m &= (s.flux / s.err) >= self.cfg.min_snr
        return m

    def segments(self, mask):
        """Fitted segments with enough pixels: list of (name, boolean array, ref_wave)."""
        w = self.spec.wave
        segs = []
        for name, lo, hi in self.cfg.segments:
            sm = mask & (w >= lo) & (w < hi)
            if sm.sum() >= self.cfg.min_pixels:
                segs.append((name, sm, DEFAULT_REF.get(name, 0.5 * (lo + hi))))
        return segs

    # ---- main entry -----------------------------------------------------
    def run(self):
        mask = self.fit_mask()
        segs = self.segments(mask)
        if not segs:
            raise ValueError("No wavelength segment has >= %d usable pixels." % self.cfg.min_pixels)
        mask = np.zeros_like(mask)
        for _, sm, _ in segs:
            mask |= sm
        if self.cfg.veiling == "none":
            vsegs = []
        elif self.cfg.veiling == "single":
            ref = self.cfg.ref_wave or segs[0][2]
            vsegs = [("all", mask.copy(), ref)]
        else:
            vsegs = segs
        cube = _chi2_cube(self.spec.flux[mask], self.spec.err[mask], self.T_all[:, mask],
                          self.alam[mask], self.cfg.av_grid,
                          [sm[mask] for _, sm, _ in vsegs])
        tref = np.array([[_ref_flux(self.grid, i, ref) for _, _, ref in vsegs] for i in range(self.teff.size)])
        tref = tref.reshape(self.teff.size, len(vsegs))
        res = FitResult(self, mask, segs, vsegs, cube, tref)
        # index-based first guess on the observed spectrum
        try:
            av0 = self.cfg.initial_av if self.cfg.initial_av is not None else res.av
            res.index_teff, res.index_dist, res.indices = index_first_guess(
                self.spec.wave, self.spec.flux, self.spec.err, self.T_all, self.teff, av=av0, rv=self.cfg.rv)
        except Exception as exc:  # indices are diagnostic only
            warnings.warn("Index measurement failed: %s" % exc)
        return res


def _ref_flux(grid, i, lam, half=0.003):
    m = np.abs(grid.wave - lam) <= half
    return float(np.median(grid.flux[i, m]))


def _chi2_cube(y, sig, T, alam, avs, seg_masks, chunk=64):
    """chi^2(Teff, A_V) minimised over C >= 0 and E_j >= 0 (exact NNLS by active sets)."""
    n_t, n = T.shape
    k = len(seg_masks)
    w = 1.0 / sig ** 2
    S = np.array(seg_masks, float).reshape(k, n)
    yy = np.sum(w * y * y)
    n_av = avs.size
    A00 = np.empty((n_t, n_av))
    r0 = np.empty((n_t, n_av))
    A0j = np.empty((n_t, n_av, k))
    Ajj = np.empty((n_av, k))
    rj = np.empty((n_av, k))
    V1 = w * T * T                                   # (n_t, n)
    V3 = w * y * T                                   # (n_t, n)
    V2 = (w * T)[:, None, :] * S[None, :, :]         # (n_t, k, n)
    for a0 in range(0, n_av, chunk):
        sl = slice(a0, min(a0 + chunk, n_av))
        E = 10.0 ** (-0.4 * avs[sl, None] * alam[None, :])   # (c, n)
        E2 = E * E
        A00[:, sl] = V1 @ E2.T
        r0[:, sl] = V3 @ E.T
        if k:
            A0j[:, sl, :] = np.einsum("tkn,cn->tck", V2, E2)
            Ajj[sl] = E2 @ (w * S).T
            rj[sl] = E @ (w * y * S).T
    best_chi = np.full((n_t, n_av), np.inf)
    best_theta = np.zeros((n_t, n_av, 1 + k))
    for size in range(k + 1):
        for active in itertools.combinations(range(k), size):
            d = 1 + size
            M = np.zeros((n_t, n_av, d, d))
            b = np.zeros((n_t, n_av, d))
            M[..., 0, 0] = A00
            b[..., 0] = r0
            for q, j in enumerate(active, start=1):
                M[..., 0, q] = M[..., q, 0] = A0j[..., j]
                M[..., q, q] = Ajj[None, :, j]
                b[..., q] = rj[None, :, j]
            M[..., range(d), range(d)] *= (1 + 1e-12)
            try:
                th = np.linalg.solve(M, b[..., None])[..., 0]
            except np.linalg.LinAlgError:
                continue
            chi = yy - np.sum(th * b, axis=-1)
            ok = (th[..., 0] > 0) & np.all(th[..., 1:] >= 0, axis=-1)
            better = ok & (chi < best_chi)
            best_chi[better] = chi[better]
            full = np.zeros((n_t, n_av, 1 + k))
            full[..., 0] = th[..., 0]
            for q, j in enumerate(active, start=1):
                full[..., 1 + j] = th[..., q]
            best_theta[better] = full[better]
    return {"chi2": np.maximum(best_chi, 0.0), "theta": best_theta, "n": n}


class FitResult(object):
    """Outcome of a fit; also the state object the eye-check GUI edits."""

    def __init__(self, fitter, mask, segs, vsegs, cube, tref):
        self.fitter = fitter
        self.spec = fitter.spec
        self.mask = mask
        self.segs = segs
        self.vsegs = vsegs
        self.chi2 = cube["chi2"]
        self.theta = cube["theta"]
        self.n = cube["n"]
        self.k = len(vsegs)
        self.m = 3 + self.k                     # Teff, A_V, C, r_j
        self.tref = tref
        self.teff = fitter.teff
        self.av_grid = fitter.cfg.av_grid
        self.index_teff, self.index_dist, self.indices = np.nan, None, {}
        self.flag = ""
        self.comment = ""
        self.method = "auto"
        self._apply_window()
        it, ia = np.unravel_index(np.argmin(self.chi2_search), self.chi2.shape)
        self.best = (int(it), int(ia))
        self.set_state(it, self.av_grid[ia])

    # ---- search window around initial values ----------------------------
    def _apply_window(self):
        cfg = self.fitter.cfg
        self.chi2_search = self.chi2.copy()
        if cfg.search_window and (cfg.initial_spt is not None or cfg.initial_av is not None):
            dsp, dav = cfg.search_window
            if cfg.initial_spt is not None:
                c0 = spt_to_code(cfg.initial_spt)
                codes = self.fitter.scale.teff_to_code(self.teff)
                bad = ~(np.abs(codes - c0) <= dsp)
                self.chi2_search[bad, :] = np.inf
            if cfg.initial_av is not None:
                bad = np.abs(self.av_grid - cfg.initial_av) > dav
                self.chi2_search[:, bad] = np.inf

    # ---- statistics ------------------------------------------------------
    @property
    def dof(self):
        return max(self.n - self.m, 1)

    @property
    def chi2_min(self):
        return float(self.chi2_search[self.best])

    @property
    def chi2nu_min(self):
        return self.chi2_min / self.dof

    def delta_chi2(self):
        """Delta chi^2 rescaled so that chi2_nu,min = 1 and for oversampled (correlated) pixels."""
        s = max(self.chi2nu_min, 1.0)
        if self.fitter.cfg.correlated_errors:
            s *= self.fitter.ppr
        return (self.chi2_search - self.chi2_min) / s

    def uncertainties(self, level=1.0):
        d = self.delta_chi2()
        prof_t = d.min(axis=1)
        prof_a = d.min(axis=0)
        t_ok = self.teff[prof_t <= level]
        a_ok = self.av_grid[prof_a <= level]
        dt = np.abs(np.diff(self.teff)).min() / 2.0
        da = np.diff(self.av_grid).min() / 2.0 if self.av_grid.size > 1 else 0.0
        inside = d <= level
        r_cells = self.r_cube()[inside] if self.k else np.zeros((0, 0))
        out = {
            "teff_lo": t_ok.min() - dt, "teff_hi": t_ok.max() + dt,
            "av_lo": max(a_ok.min() - da, 0.0), "av_hi": a_ok.max() + da,
        }
        for j in range(self.k):
            out["r_lo_%s" % self.vsegs[j][0]] = float(r_cells[:, j].min()) if r_cells.size else np.nan
            out["r_hi_%s" % self.vsegs[j][0]] = float(r_cells[:, j].max()) if r_cells.size else np.nan
        return out

    def r_cube(self):
        a = self.theta[..., 0]
        with np.errstate(divide="ignore", invalid="ignore"):
            return self.theta[..., 1:] / (a[..., None] * self.tref[:, None, :])

    def refined_teff(self):
        """Parabolic refinement of the Teff profile around the grid minimum."""
        prof = self.chi2_search.min(axis=1)
        i = self.best[0]
        if 0 < i < self.teff.size - 1 and np.all(np.isfinite(prof[i - 1:i + 2])):
            x, y = self.teff[i - 1:i + 2], prof[i - 1:i + 2]
            c = np.polyfit(x, y, 2)
            if c[0] > 0:
                t = -c[1] / (2 * c[0])
                if x.min() <= t <= x.max():
                    return float(t)
        return float(self.teff[i])

    # ---- current state (edited by the GUI) -------------------------------
    def set_state(self, it, av, r=None):
        """Select template index ``it`` and A_V; C and veiling are re-solved unless ``r`` is fixed."""
        self.it = int(np.clip(it, 0, self.teff.size - 1))
        self.av = float(av)
        self.r_fixed = None if r is None else np.atleast_1d(np.asarray(r, float))
        self.C, self.E, self.chi2_cur = self.solve_at(self.it, self.av, self.r_fixed)

    def solve_at(self, it, av, r=None):
        sel = self.mask
        y, s = self.spec.flux[sel], self.spec.err[sel]
        w = 1 / s ** 2
        T = self.fitter.T_all[it, sel]
        e = 10 ** (-0.4 * av * self.fitter.alam[sel])
        S = np.array([vs[1][sel] for vs in self.vsegs], float).reshape(self.k, int(sel.sum()))
        if r is not None and self.k:
            r = np.resize(r, self.k)
            g = (T + np.sum((r * self.tref[it])[:, None] * S, axis=0)) * e
            C = max(np.sum(w * y * g) / np.sum(w * g * g), 0.0)
            E = r * self.tref[it]
        else:
            cols = [T * e] + [Sj * e for Sj in S]
            best = (np.inf, 0.0, np.zeros(self.k))
            for size in range(self.k + 1):
                for active in itertools.combinations(range(self.k), size):
                    X = np.array([cols[0]] + [cols[1 + j] for j in active]).T
                    th, *_ = np.linalg.lstsq(X * np.sqrt(w)[:, None], y * np.sqrt(w), rcond=None)
                    if th[0] <= 0 or np.any(th[1:] < 0):
                        continue
                    chi = np.sum(w * (y - X @ th) ** 2)
                    if chi < best[0]:
                        Ev = np.zeros(self.k)
                        for q, j in enumerate(active, start=1):
                            Ev[j] = th[q] / th[0]
                        best = (chi, th[0], Ev)
            C, E = best[1], best[2]
        chi = np.sum(w * (y - self.model(it, av, C, E)[sel]) ** 2)
        return float(C), np.asarray(E, float), float(chi)

    @property
    def r(self):
        return self.E / self.tref[self.it] if self.k else np.zeros(0)

    def excess(self, it=None, E=None):
        """Flat excess E_j evaluated on all pixels (zero outside veiling segments)."""
        E = self.E if E is None else E
        ex = np.zeros(self.spec.wave.size)
        w = self.spec.wave
        for j, (name, sm, ref) in enumerate(self.vsegs):
            if name == "all":
                lo = min(s[1].nonzero()[0].min() for s in self.segs)
                hi = max(s[1].nonzero()[0].max() for s in self.segs)
                ex[lo:hi + 1] = E[j]
            else:
                lo, hi = [(a, b) for n, a, b in self.fitter.cfg.segments if n == name][0]
                ex[(w >= lo) & (w < hi)] = E[j]
        return ex

    def model(self, it=None, av=None, C=None, E=None):
        """Reddened, veiled model on every observed pixel."""
        it = self.it if it is None else it
        av = self.av if av is None else av
        C = self.C if C is None else C
        E = self.E if E is None else E
        return C * (self.fitter.T_all[it] + self.excess(it, E)) * 10 ** (-0.4 * av * self.fitter.alam)

    def corrected(self):
        """Observed spectrum dereddened, rescaled by 1/C and with the excess removed."""
        return self.spec.flux * 10 ** (0.4 * self.av * self.fitter.alam) / self.C - self.excess()

    def dereddened(self):
        return self.spec.flux * 10 ** (0.4 * self.av * self.fitter.alam) / self.C

    def template(self, it=None):
        return self.fitter.T_all[self.it if it is None else it]

    # ---- reporting -------------------------------------------------------
    @property
    def teff_cur(self):
        if self.method == "auto" and self.it == self.best[0]:
            return self.refined_teff()
        return float(self.teff[self.it])

    @property
    def spt(self):
        return self.fitter.scale.teff_to_spt(self.teff_cur)

    def summary(self):
        unc = self.uncertainties()
        sc = self.fitter.scale
        lo, hi = sc.teff_to_code(unc["teff_hi"]), sc.teff_to_code(unc["teff_lo"])
        c = sc.teff_to_code(self.teff_cur)
        spt_err = max(0.25, 0.5 * (hi - lo)) if np.isfinite(lo) and np.isfinite(hi) else np.nan
        row = {
            "name": self.spec.name, "instrument": self.spec.instrument, "R": self.fitter.R,
            "segments": "+".join(s[0] for s in self.segs), "veiling_mode": self.fitter.cfg.veiling,
            "SpT": self.spt, "SpT_code": round(float(c), 3) if np.isfinite(c) else np.nan,
            "SpT_err": round(float(spt_err), 2) if np.isfinite(spt_err) else np.nan,
            "Teff": round(self.teff_cur, 1), "Teff_lo": unc["teff_lo"], "Teff_hi": unc["teff_hi"],
            "Av": round(self.av, 2), "Av_lo": round(unc["av_lo"], 2), "Av_hi": round(unc["av_hi"], 2),
            "chi2nu": round(self.chi2_cur / self.dof, 3), "n_pix": int(self.n),
            "index_SpT": sc.teff_to_spt(self.index_teff) if np.isfinite(self.index_teff) else "--",
            "initial_SpT": self.fitter.cfg.initial_spt or "", "initial_Av": self.fitter.cfg.initial_av if self.fitter.cfg.initial_av is not None else "",
            "method": self.method, "flag": self.flag, "comment": self.comment,
        }
        for j, (name, _, ref) in enumerate(self.vsegs):
            row["r_%s" % name] = round(float(self.r[j]), 3)
            row["r_%s_lo" % name] = round(unc["r_lo_%s" % name], 3)
            row["r_%s_hi" % name] = round(unc["r_hi_%s" % name], 3)
            row["lam_ref_%s" % name] = ref
        return row


def fit_spectrum(spectrum, grid, resolving_power, **kwargs):
    """Convenience wrapper: ``fit_spectrum(spec, grid, 3000, veiling='per-band')``."""
    return Fitter(spectrum, grid, resolving_power, FitConfig(**kwargs)).run()
