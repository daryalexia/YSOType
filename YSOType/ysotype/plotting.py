"""Plotting helpers shared by the eye-check GUI and the publication figures."""
import numpy as np

from .spectrum import EMISSION_LINES, TELLURIC_BANDS, ranges_to_mask

COLORS = {"observed": "#2ca02c", "corrected": "#ff7f0e", "model": "#1f77b4",
          "telluric": "0.88", "line": "#d9c8f0", "user": "#f4a6a6", "neighbour": "0.55"}

# (label, wavelength in micron) -- drawn when inside the plotted range
FEATURES = [
    ("TiO", 0.7053), ("VO", 0.7400), ("TiO", 0.7666), ("VO", 0.7900), ("Na I", 0.8190),
    ("TiO", 0.8432), ("Ca II", 0.8542), ("FeH", 0.9896), ("Na I", 1.1383), ("K I", 1.1690),
    ("K I", 1.1773), ("FeH", 1.1970), ("K I", 1.2432), ("K I", 1.2522), ("H$_2$O", 1.3400),
    ("H$_2$O", 1.8000), ("Na I", 2.2062), ("CO", 2.2935), ("CO", 2.3227), ("H$_2$O", 2.4000),
    ("H$\\alpha$", 0.65628),
]

# Zoom windows for the eye-check (title, lo, hi)
ZOOMS = [
    ("TiO 7140", 0.700, 0.730), ("VO 7445", 0.732, 0.760), ("Na I 8190", 0.810, 0.826),
    ("TiO 8432 / Ca II", 0.836, 0.870), ("K I 1.17", 1.163, 1.185), ("K I 1.25", 1.238, 1.258),
    ("H-band shape", 1.48, 1.78), ("Na I / CO (K)", 2.19, 2.33),
]


def norm_window(res):
    """Reference wavelength used to normalise displays (0.75 um optical, 1.25 um NIR)."""
    w = res.spec.wave
    for lam in (0.75, 1.25, 1.65, 2.20):
        if w.min() < lam - 0.01 and w.max() > lam + 0.01:
            return lam
    return float(np.median(w))


def _median_near(wave, flux, lam, half=0.005, good=None):
    m = np.abs(wave - lam) <= half
    if good is not None:
        m &= good
    v = np.nanmedian(flux[m]) if m.sum() > 2 else np.nanmedian(flux)
    return v if v and np.isfinite(v) else 1.0


def shade_masks(ax, res, user_masks=(), lines=True):
    lo, hi = res.spec.wave.min(), res.spec.wave.max()
    for a, b in TELLURIC_BANDS:
        if b > lo and a < hi:
            ax.axvspan(max(a, lo), min(b, hi), color=COLORS["telluric"], zorder=0, lw=0)
    if lines:
        from .spectrum import line_halfwidth
        for lam in EMISSION_LINES.values():
            if lo < lam < hi:
                hw = float(line_halfwidth(lam, res.fitter.R, res.fitter.cfg.line_velocity))
                ax.axvspan(lam - hw, lam + hw, color=COLORS["line"], zorder=0, lw=0)
    for a, b in user_masks:
        ax.axvspan(a, b, facecolor=COLORS["user"], alpha=0.6, hatch="//", edgecolor="r", zorder=0, lw=0)


def label_features(ax, xlim=None, y=0.97, fontsize=8):
    x0, x1 = xlim or ax.get_xlim()
    for lab, lam in FEATURES:
        if x0 < lam < x1:
            ax.axvline(lam, color="0.6", lw=0.6, ls=":", zorder=1)
            ax.text(lam, y, lab, transform=ax.get_xaxis_transform(), rotation=90, ha="right",
                    va="top", fontsize=fontsize, color="0.35")


def spectra_to_plot(res, frame="corrected"):
    """Return dict of normalised arrays for plotting in the chosen frame."""
    w = res.spec.wave
    lam = norm_window(res)
    good = res.spec.good
    if frame == "corrected":
        T = res.template()
        n = _median_near(w, T, lam)
        obs = res.spec.flux
        corr = res.corrected() / n
        m = res.mask & np.isfinite(obs)
        # scale the (reddened) observed spectrum so that it stays inside the panel
        o98 = np.nanpercentile(obs[m], 98) if m.any() else np.nanmax(obs)
        c98 = np.nanpercentile(corr[m], 98) if m.any() else 1.0
        return {"wave": w, "observed": obs * (c98 / o98 if o98 else 1.0),
                "corrected": hide_tellurics(w, corr), "model": T / n, "norm_lam": lam,
                "ylabel": r"$F_\lambda$ / $F_\lambda$(%.2f $\mu$m)" % lam}
    mod = res.model()
    n = _median_near(w, mod, lam)
    phot = res.C * res.template() * 10 ** (-0.4 * res.av * res.fitter.alam)
    return {"wave": w, "observed": res.spec.flux / n, "model": mod / n, "photosphere": phot / n,
            "norm_lam": lam, "ylabel": r"$F_\lambda$ (normalised)"}


def hide_tellurics(wave, y):
    """Blank telluric gaps in displayed (dereddened) spectra, where noise is amplified."""
    return np.where(ranges_to_mask(wave, TELLURIC_BANDS), np.nan, y)


def residuals(res):
    r = (res.spec.flux - res.model()) / res.spec.err
    return np.where(res.mask, r, np.nan), np.where(res.mask, np.nan, r)


def publication_figure(res, path=None, frame="corrected", figsize=(7.2, 4.0), title=None,
                       show_residuals=True, dpi=300, smooth=None):
    """Clean, journal-ready figure of the adopted solution."""
    import matplotlib.pyplot as plt
    if show_residuals:
        fig, (ax, axr) = plt.subplots(2, 1, figsize=figsize, sharex=True,
                                      gridspec_kw={"height_ratios": [3.2, 1], "hspace": 0.05})
    else:
        fig, ax = plt.subplots(figsize=figsize)
        axr = None
    d = spectra_to_plot(res, frame)
    w = d["wave"]

    def sm(y):
        if not smooth:
            return y
        from scipy.ndimage import uniform_filter1d
        return uniform_filter1d(np.nan_to_num(y), int(smooth))

    shade_masks(ax, res, res.fitter.user_masks, lines=False)
    if frame == "corrected":
        ax.plot(w, sm(d["observed"]), color=COLORS["observed"], lw=0.6, alpha=0.8, label="Observed (scaled)")
        lab = "Dereddened" + (" + de-veiled" if res.k and np.any(res.r > 0) else "")
        ax.plot(w, sm(d["corrected"]), color=COLORS["corrected"], lw=0.7, label=lab)
        ax.plot(w, d["model"], color=COLORS["model"], lw=0.9, label="%s %s" % (res.fitter.grid.label, res.spt))
    else:
        ax.plot(w, sm(d["observed"]), color=COLORS["observed"], lw=0.6, label="Observed")
        ax.plot(w, d["model"], color=COLORS["model"], lw=0.9, label="Model")
        ax.plot(w, d["photosphere"], color=COLORS["model"], lw=0.7, ls="--", label="Photosphere")
    ax.set_xlim(w.min(), w.max())
    yv = np.concatenate([d["model"][res.mask], d.get("corrected", d["model"])[res.mask]])
    lo, hi = np.nanpercentile(yv, [0.5, 99.5])
    ax.set_ylim(min(0, lo) - 0.05 * (hi - lo), hi + 0.35 * (hi - lo))
    label_features(ax)
    ax.set_ylabel(d["ylabel"])
    rtxt = ", ".join(r"$r_{%s}$=%.2f" % (n, v) for (n, _, _), v in zip(res.vsegs, res.r)) if res.k else ""
    txt = r"%s   $A_V$ = %.1f mag   %s" % (res.spt, res.av, rtxt)
    ax.text(0.01, 0.03, txt, transform=ax.transAxes, fontsize=9, ha="left", va="bottom")
    ax.legend(loc="lower right", fontsize=7, frameon=False, ncol=3)
    ax.set_title(title or "%s  (%s, R~%d)" % (res.spec.name, res.spec.instrument, res.fitter.R), fontsize=10)
    if axr is not None:
        rin, rout = residuals(res)
        axr.plot(w, rin, color="k", lw=0.4)
        axr.plot(w, rout, color="0.75", lw=0.4)
        axr.axhline(0, color=COLORS["model"], lw=0.7)
        axr.set_ylim(-6, 6)
        axr.set_ylabel(r"$\Delta/\sigma$")
        axr.set_xlabel(r"Wavelength ($\mu$m)")
    else:
        ax.set_xlabel(r"Wavelength ($\mu$m)")
    fig.subplots_adjust(left=0.10, right=0.98, top=0.92, bottom=0.12)
    if path:
        fig.savefig(path, dpi=dpi)
    return fig
