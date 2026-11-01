"""Interactive visual verification of a YSOType fit (matplotlib only).

Layout
------
  +----------------------------------------+-------------------+
  | spectrum (corrected or observed frame) | chi^2 map         |
  | residuals                              | (A_V vs Teff)     |
  +-------------+------------+-------------+ SpT profile       |
  | zoom 1      | zoom 2     | zoom 3      | solution panel    |
  +-------------+------------+-------------+-------------------+
  | sliders: SpT, A_V, r      | buttons                        |

Keys:  <- / ->  hotter / cooler template     up / down  A_V +-0.5 (shift: +-5)
       [ / ]    veiling -+0.05 (locks r)     l   lock / unlock veiling
       b        back to best fit             d   corrected <-> observed frame
       n        neighbouring templates       z   next zoom panels
       m        mask mode (drag = mask, right-click = unmask)   u  undo mask
       f        refit with current masks     g   cycle flag
       s        save publication figure      a / Enter  accept     q  quit
Clicking on the chi^2 map jumps to that (Teff, A_V).
"""
import numpy as np

from . import plotting as P

_KEYS = ["left", "right", "up", "down", "shift+up", "shift+down", "[", "]", "l", "b", "d",
         "n", "z", "m", "u", "f", "g", "s", "a", "enter", "q"]
FLAGS = ["", "veiled", "low-S/N", "uncertain", "check"]


def _free_keymaps(plt):
    saved = {}
    for k in list(plt.rcParams.keys()):
        if k.startswith("keymap."):
            saved[k] = list(plt.rcParams[k])
            plt.rcParams[k] = [x for x in plt.rcParams[k] if x not in _KEYS]
    return saved


class EyeCheck(object):
    """``EyeCheck(result).run()`` -> (result, accepted: bool)."""

    def __init__(self, result, on_accept=None, figure_path=None, interactive=True):
        import matplotlib
        import matplotlib.pyplot as plt
        self.plt = plt
        self.res = result
        self.on_accept = on_accept
        self.figure_path = figure_path
        self.frame = "corrected"
        self.show_neighbours = False
        self.mask_mode = False
        self.accepted = False
        self.zoom_page = 0
        self._busy = False
        self.status = "Automatic best fit. Inspect, adjust, then press 'a' to accept."
        if interactive:
            self._saved_keymaps = _free_keymaps(plt)
        else:
            self._saved_keymaps = {}
        self.fig = plt.figure(figsize=(16, 9))
        try:
            self.fig.canvas.manager.set_window_title("YSOType eye-check: %s" % result.spec.name)
        except Exception:
            pass
        self._build_axes()
        self._build_widgets()
        self._zoom_windows = [z for z in P.ZOOMS
                              if np.sum((result.spec.wave > z[1]) & (result.spec.wave < z[2])) > 10]
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        self.fig.canvas.mpl_connect("button_press_event", self._on_click)
        self.redraw(full=True)

    # ------------------------------------------------------------------ UI
    def _build_axes(self):
        f = self.fig
        self.ax = f.add_axes([0.045, 0.50, 0.62, 0.44])
        self.axr = f.add_axes([0.045, 0.40, 0.62, 0.10], sharex=self.ax)
        self.axz = [f.add_axes([0.045 + i * 0.215, 0.165, 0.19, 0.155]) for i in range(3)]
        self.axm = f.add_axes([0.725, 0.58, 0.255, 0.36])
        self.axp = f.add_axes([0.725, 0.40, 0.255, 0.12])
        self.axi = f.add_axes([0.725, 0.115, 0.255, 0.245])
        self.axi.axis("off")

    def _build_widgets(self):
        from matplotlib.widgets import Button, Slider
        f, res = self.fig, self.res
        n_t = res.teff.size
        self.s_spt = Slider(f.add_axes([0.08, 0.075, 0.36, 0.018]), "SpT", 0, n_t - 1,
                            valinit=res.it, valstep=1)
        self.s_av = Slider(f.add_axes([0.08, 0.047, 0.36, 0.018]), "A$_V$", res.av_grid[0], res.av_grid[-1],
                           valinit=res.av, valstep=float(res.av_grid[1] - res.av_grid[0]) if res.av_grid.size > 1 else None)
        self.s_r = Slider(f.add_axes([0.08, 0.019, 0.36, 0.018]), "r", 0.0, 5.0,
                          valinit=float(res.r[0]) if res.k else 0.0, valstep=0.01)
        if not res.k:
            self.s_r.ax.set_visible(False)
        self.s_spt.on_changed(self._spt_changed)
        self.s_av.on_changed(self._av_changed)
        self.s_r.on_changed(self._r_changed)
        names = [("Best fit", self.best), ("Lock r", self.toggle_lock), ("Mask", self.toggle_mask),
                 ("Refit", self.refit), ("Frame", self.toggle_frame), ("Flag", self.cycle_flag),
                 ("Save fig", self.save_figure), ("Accept", self.accept)]
        self.buttons = []
        for i, (lab, fn) in enumerate(names):
            r, c = divmod(i, 4)
            b = Button(f.add_axes([0.55 + c * 0.108, 0.065 - r * 0.045, 0.1, 0.035]), lab,
                       color="#d8ecd8" if lab == "Accept" else "0.92", hovercolor="0.8")
            b.on_clicked(lambda evt, fn=fn: fn())
            self.buttons.append(b)
        self._span = None

    def _make_span(self):
        from matplotlib.widgets import SpanSelector
        for kw in ({"props": dict(alpha=0.3, facecolor="red")}, {"rectprops": dict(alpha=0.3, facecolor="red")}, {}):
            try:
                return SpanSelector(self.ax, self._on_span, "horizontal", useblit=True, **kw)
            except TypeError:
                continue

    # ------------------------------------------------------------- drawing
    def redraw(self, full=False):
        res = self.res
        self._sync_sliders()
        self._draw_spectrum()
        self._draw_zooms()
        if full or not hasattr(self, "_map_img"):
            self._draw_map()
        self._draw_markers()
        self._draw_info()
        self.fig.canvas.draw_idle()

    def _draw_spectrum(self):
        res, ax, axr = self.res, self.ax, self.axr
        xl = ax.get_xlim() if ax.lines else None
        ax.cla()
        axr.cla()
        d = P.spectra_to_plot(res, self.frame)
        w = d["wave"]
        P.shade_masks(ax, res, res.fitter.user_masks)
        if self.frame == "corrected":
            ax.plot(w, d["observed"], color=P.COLORS["observed"], lw=0.5, alpha=0.55, label="observed (scaled)")
            if self.show_neighbours:
                for it, ls in self._neighbours():
                    T = res.template(it)
                    n = P._median_near(w, T, d["norm_lam"])
                    ax.plot(w, T / n, color=P.COLORS["neighbour"], lw=0.7, ls=ls,
                            label=res.fitter.scale.teff_to_spt(res.teff[it]))
            ax.plot(w, d["corrected"], color=P.COLORS["corrected"], lw=0.6, label="dereddened%s" % (" + de-veiled" if res.k else ""))
            ax.plot(w, d["model"], color=P.COLORS["model"], lw=0.9, label="template %s" % res.spt)
            yv = np.concatenate([d["model"][res.mask], d["corrected"][res.mask]])
        else:
            ax.plot(w, d["observed"], color=P.COLORS["observed"], lw=0.5, label="observed")
            ax.plot(w, d["model"], color=P.COLORS["model"], lw=0.9, label="model")
            ax.plot(w, d["photosphere"], color=P.COLORS["model"], lw=0.7, ls="--", label="photosphere")
            yv = np.concatenate([d["model"][res.mask], d["observed"][res.mask]])
        lo, hi = np.nanpercentile(yv, [0.5, 99.5])
        ax.set_ylim(min(lo, 0) - 0.05 * (hi - lo), hi + 0.3 * (hi - lo))
        ax.set_xlim(xl if xl else (w.min(), w.max()))
        P.label_features(ax, (w.min(), w.max()), fontsize=7)
        ax.legend(loc="lower right", fontsize=8, ncol=5, frameon=False)
        ax.set_ylabel(d["ylabel"])
        ax.set_title("%s  |  %s  R=%d  |  %s frame%s" % (
            res.spec.name, res.spec.instrument or "?", res.fitter.R, self.frame,
            "  |  MASK MODE: drag to mask, right-click to unmask" if self.mask_mode else ""),
            fontsize=10, loc="left", color="r" if self.mask_mode else "k")
        rin, rout = P.residuals(res)
        axr.plot(w, rin, color="k", lw=0.4)
        axr.plot(w, rout, color="0.75", lw=0.4)
        axr.axhline(0, color=P.COLORS["model"], lw=0.7)
        axr.set_ylim(-6, 6)
        axr.set_ylabel(r"$\Delta/\sigma$")
        axr.set_xlabel(r"Wavelength ($\mu$m)")
        self.ax.tick_params(labelbottom=False)
        self._dplot = d

    def _neighbours(self):
        """Templates one subclass hotter and cooler than the current one."""
        res = self.res
        codes = res.fitter.scale.teff_to_code(res.teff)
        c0 = codes[res.it]
        out = []
        for dc, ls in ((-1.0, ":"), (1.0, "--")):
            if np.isfinite(c0):
                j = int(np.nanargmin(np.abs(codes - (c0 + dc))))
                if j != res.it:
                    out.append((j, ls))
        return out

    def _draw_zooms(self):
        d, res = self._dplot, self.res
        w = d["wave"]
        wins = self._zoom_windows
        for i, ax in enumerate(self.axz):
            ax.cla()
            k = self.zoom_page * 3 + i
            if not wins or k >= len(wins):
                ax.set_visible(False)
                continue
            ax.set_visible(True)
            title, lo, hi = wins[k]
            m = (w >= lo) & (w <= hi)
            P.shade_masks(ax, res, res.fitter.user_masks)
            if self.frame == "corrected":
                ys = [d["corrected"][m], d["model"][m]]
                ax.plot(w[m], d["corrected"][m], color=P.COLORS["corrected"], lw=0.8)
                if self.show_neighbours:
                    for it, ls in self._neighbours():
                        T = res.template(it)
                        n = P._median_near(w, T, d["norm_lam"])
                        ax.plot(w[m], T[m] / n, color=P.COLORS["neighbour"], lw=0.7, ls=ls)
                ax.plot(w[m], d["model"][m], color=P.COLORS["model"], lw=1.0)
            else:
                ys = [d["observed"][m], d["model"][m]]
                ax.plot(w[m], d["observed"][m], color=P.COLORS["observed"], lw=0.8)
                ax.plot(w[m], d["model"][m], color=P.COLORS["model"], lw=1.0)
            yy = np.concatenate(ys)
            yy = yy[np.isfinite(yy)]
            if yy.size:
                a, b = np.percentile(yy, [1, 99])
                ax.set_ylim(a - 0.1 * (b - a), b + 0.25 * (b - a))
            ax.set_xlim(lo, hi)
            P.label_features(ax, (lo, hi), fontsize=6)
            ax.set_title(title, fontsize=9)
            ax.tick_params(labelsize=7)

    def _draw_map(self):
        res, ax = self.res, self.axm
        ax.cla()
        D = res.delta_chi2()
        Dp = np.log10(1.0 + np.where(np.isfinite(D), D, np.nan))
        self._map_img = ax.pcolormesh(res.av_grid, res.teff, Dp, cmap="viridis_r", shading="nearest")
        try:
            ax.contour(res.av_grid, res.teff, np.where(np.isfinite(D), D, 1e9), levels=[1, 4, 9],
                       colors=["w", "w", "w"], linewidths=[1.2, 0.8, 0.5])
        except Exception:
            pass
        fin = np.isfinite(D) & (D < 400)
        if fin.any():
            ia = np.where(fin.any(axis=0))[0]
            it = np.where(fin.any(axis=1))[0]
            pad_a = max(2.0, 0.15 * (res.av_grid[ia[-1]] - res.av_grid[ia[0]]))
            x0, x1 = res.av_grid[ia[0]] - pad_a, res.av_grid[ia[-1]] + pad_a
            y0, y1 = res.teff[it].min() - 300, res.teff[it].max() + 300
            cfg = res.fitter.cfg
            if cfg.initial_av is not None:
                x0, x1 = min(x0, cfg.initial_av - 1), max(x1, cfg.initial_av + 1)
            if cfg.initial_spt is not None:
                t0 = res.fitter.scale.spt_to_teff(cfg.initial_spt)
                if np.isfinite(t0):
                    y0, y1 = min(y0, t0 - 150), max(y1, t0 + 150)
            ax.set_xlim(max(res.av_grid[0], x0), min(res.av_grid[-1], x1))
            ax.set_ylim(y0, y1)
        ax.set_xlabel(r"$A_V$ (mag)", fontsize=9)
        ax.set_ylabel(r"$T_{\rm eff}$ (K)", fontsize=9)
        ax.set_title(r"log$_{10}$(1+$\Delta\chi^2$)  contours: 1, 4, 9   (click to jump)", fontsize=8)
        ax.tick_params(labelsize=7)
        sec = ax.secondary_yaxis("right", functions=(lambda t: t, lambda t: t)) if hasattr(ax, "secondary_yaxis") else None
        if sec is not None:
            ticks = [t for t in res.teff[::2] if ax.get_ylim()[0] <= t <= ax.get_ylim()[1]]
            sec.set_yticks(ticks)
            sec.set_yticklabels([res.fitter.scale.teff_to_spt(t, step=0.5) for t in ticks], fontsize=6)
        # SpT profile
        axp = self.axp
        axp.cla()
        prof = D.min(axis=1)
        codes = res.fitter.scale.teff_to_code(res.teff)
        ok = np.isfinite(prof) & np.isfinite(codes)
        axp.plot(codes[ok], prof[ok], "o-", ms=3, color="k", lw=0.8)
        axp.axhline(1, color="0.6", ls=":", lw=0.8)
        axp.set_yscale("symlog", linthresh=1) if hasattr(axp, "set_yscale") else None
        lo_c = np.nanmin(codes[ok & (prof < 100)]) - 1.5 if np.any(ok & (prof < 100)) else np.nanmin(codes[ok])
        hi_c = np.nanmax(codes[ok & (prof < 100)]) + 1.5 if np.any(ok & (prof < 100)) else np.nanmax(codes[ok])
        axp.set_xlim(lo_c, hi_c)
        axp.set_ylim(bottom=0)
        ticks = np.arange(np.ceil(lo_c), np.floor(hi_c) + 1)
        axp.set_xticks(ticks)
        from .spt import code_to_spt
        axp.set_xticklabels([code_to_spt(t, 1) for t in ticks], fontsize=7)
        axp.set_ylabel(r"$\Delta\chi^2_{\rm min}$", fontsize=8)
        axp.tick_params(labelsize=7)

    def _draw_markers(self):
        res = self.res
        for a in getattr(self, "_markers", []):
            try:
                a.remove()
            except Exception:
                pass
        mk = []
        ib = res.best
        mk += self.axm.plot(res.av_grid[ib[1]], res.teff[ib[0]], marker="*", ms=14, color="w", mec="k", ls="")
        mk += self.axm.plot(res.av, res.teff[res.it], marker="+", ms=16, mew=2, color="r", ls="")
        cfg = res.fitter.cfg
        if cfg.initial_spt is not None or cfg.initial_av is not None:
            t0 = res.fitter.scale.spt_to_teff(cfg.initial_spt) if cfg.initial_spt is not None else res.teff[res.it]
            a0 = cfg.initial_av if cfg.initial_av is not None else res.av
            mk += self.axm.plot(a0, t0, marker="x", ms=10, mew=2, color="c", ls="")
        c = res.fitter.scale.teff_to_code(res.teff[res.it])
        if np.isfinite(c):
            mk.append(self.axp.axvline(c, color="r", lw=1))
        self._markers = mk

    def _draw_info(self):
        res, ax = self.res, self.axi
        ax.cla()
        ax.axis("off")
        row = res.summary()
        lines = [
            ("SpT", "%s  (± %.2f)" % (row["SpT"], row["SpT_err"])),
            ("Teff", "%d K  [%d, %d]" % (row["Teff"], row["Teff_lo"], row["Teff_hi"])),
            ("A_V", "%.2f mag  [%.2f, %.2f]" % (row["Av"], row["Av_lo"], row["Av_hi"])),
        ]
        for name, _, ref in res.vsegs:
            lines.append(("r_%s" % name, "%.2f  [%.2f, %.2f]  @%.2f um%s" % (
                row["r_%s" % name], row["r_%s_lo" % name], row["r_%s_hi" % name], ref,
                "  LOCKED" if res.r_fixed is not None else "")))
        lines += [
            ("chi2_nu", "%.2f   (n=%d, m=%d)" % (row["chi2nu"], res.n, res.m)),
            ("index SpT", str(row["index_SpT"])),
            ("initial", "%s / %s" % (row["initial_SpT"] or "-", row["initial_Av"] if row["initial_Av"] != "" else "-")),
            ("segments", row["segments"]),
            ("method", row["method"] + ("   FLAG: %s" % res.flag if res.flag else "")),
        ]
        y = 1.0
        ax.text(0, y, "Solution", fontsize=11, weight="bold", transform=ax.transAxes, va="top")
        y -= 0.09
        for k, v in lines:
            ax.text(0.0, y, k, fontsize=8.5, transform=ax.transAxes, va="top", color="0.35")
            ax.text(0.30, y, v, fontsize=8.5, transform=ax.transAxes, va="top", family="monospace")
            y -= 0.068
        ax.text(0, y - 0.02, self.status, fontsize=8, transform=ax.transAxes, va="top", color="#a33", wrap=True)

    # --------------------------------------------------------- state sync
    def _sync_sliders(self):
        res = self.res
        self._busy = True
        try:
            if int(self.s_spt.val) != res.it:
                self.s_spt.set_val(res.it)
            if abs(self.s_av.val - res.av) > 1e-9:
                self.s_av.set_val(res.av)
            if res.k and abs(self.s_r.val - float(res.r[0])) > 1e-3:
                self.s_r.set_val(min(float(res.r[0]), self.s_r.valmax))
            self.s_spt.valtext.set_text("%s (%d K)" % (res.fitter.scale.teff_to_spt(res.teff[res.it]), res.teff[res.it]))
        finally:
            self._busy = False

    def _set(self, it=None, av=None, r="keep"):
        res = self.res
        it = res.it if it is None else it
        av = res.av if av is None else float(np.clip(av, res.av_grid[0], res.av_grid[-1]))
        rr = res.r_fixed if r == "keep" else r
        res.set_state(it, av, rr)
        res.method = "auto" if (res.it, res.av) == (res.best[0], res.av_grid[res.best[1]]) and res.r_fixed is None else "visual"
        self.redraw()

    def _spt_changed(self, val):
        if not self._busy:
            self._set(it=int(round(val)))

    def _av_changed(self, val):
        if not self._busy:
            self._set(av=val)

    def _r_changed(self, val):
        if not self._busy and self.res.k:
            self._set(r=np.full(self.res.k, float(val)))
            self.status = "Veiling locked at r=%.2f (press 'l' to free it)." % val

    # ------------------------------------------------------------ actions
    def best(self):
        it, ia = self.res.best
        self.status = "Back to the automatic best fit."
        self._set(it=it, av=self.res.av_grid[ia], r=None)

    def toggle_lock(self):
        res = self.res
        if not res.k:
            return
        if res.r_fixed is None:
            self.status = "Veiling locked at current values."
            self._set(r=res.r.copy())
        else:
            self.status = "Veiling free: re-solved for each template / A_V."
            self._set(r=None)

    def toggle_frame(self):
        self.frame = "observed" if self.frame == "corrected" else "corrected"
        self.redraw()

    def toggle_mask(self):
        self.mask_mode = not self.mask_mode
        if self.mask_mode and self._span is None:
            self._span = self._make_span()
        if self._span is not None:
            self._span.set_active(self.mask_mode)
        self.status = "Mask mode ON: drag over the spectrum; 'f' to refit." if self.mask_mode else "Mask mode OFF."
        self.redraw()

    def _on_span(self, xmin, xmax):
        if self.mask_mode and xmax > xmin:
            self.res.fitter.user_masks.append((float(xmin), float(xmax)))
            self.status = "Masked %.4f-%.4f um. Press 'f' (Refit) to update the fit." % (xmin, xmax)
            self.redraw()

    def undo_mask(self):
        if self.res.fitter.user_masks:
            a, b = self.res.fitter.user_masks.pop()
            self.status = "Removed mask %.4f-%.4f um." % (a, b)
            self.redraw()

    def refit(self):
        fl, cm = self.res.flag, self.res.comment
        self.res = self.res.fitter.run()
        self.res.flag, self.res.comment = fl, cm
        self.status = "Refitted with %d user mask(s)." % len(self.res.fitter.user_masks)
        self.redraw(full=True)

    def cycle_flag(self):
        i = FLAGS.index(self.res.flag) if self.res.flag in FLAGS else 0
        self.res.flag = FLAGS[(i + 1) % len(FLAGS)]
        self.status = "Flag: %s" % (self.res.flag or "none")
        self.redraw()

    def save_figure(self, path=None):
        path = path or self.figure_path or "%s_ysotype.pdf" % self.res.spec.name
        fig = P.publication_figure(self.res, path, frame=self.frame)
        self.plt.close(fig)
        self.status = "Saved %s" % path
        self.redraw()
        return path

    def accept(self):
        self.accepted = True
        if self.on_accept:
            self.on_accept(self.res)
        if self.figure_path:
            self.save_figure(self.figure_path)
        self.plt.close(self.fig)

    # ------------------------------------------------------------- events
    def _on_key(self, ev):
        k, res = ev.key, self.res
        if k == "left":
            self._set(it=res.it - 1)
        elif k == "right":
            self._set(it=res.it + 1)
        elif k in ("up", "down", "shift+up", "shift+down"):
            step = 5.0 if k.startswith("shift") else 0.5
            self._set(av=res.av + (step if k.endswith("up") else -step))
        elif k in ("[", "]") and res.k:
            r = np.clip(res.r + (0.05 if k == "]" else -0.05), 0, None)
            self.status = "Veiling locked (press 'l' to free it)."
            self._set(r=r)
        elif k == "l":
            self.toggle_lock()
        elif k == "b":
            self.best()
        elif k == "d":
            self.toggle_frame()
        elif k == "n":
            self.show_neighbours = not self.show_neighbours
            self.redraw()
        elif k == "z":
            npg = max(1, int(np.ceil(len(self._zoom_windows) / 3.0)))
            self.zoom_page = (self.zoom_page + 1) % npg
            self.redraw()
        elif k == "m":
            self.toggle_mask()
        elif k == "u":
            self.undo_mask()
        elif k == "f":
            self.refit()
        elif k == "g":
            self.cycle_flag()
        elif k == "s":
            self.save_figure()
        elif k in ("a", "enter"):
            self.accept()
        elif k == "q":
            self.plt.close(self.fig)

    def _on_click(self, ev):
        if ev.inaxes is self.axm and ev.button == 1 and ev.xdata is not None:
            it = int(np.argmin(np.abs(self.res.teff - ev.ydata)))
            ia = int(np.argmin(np.abs(self.res.av_grid - ev.xdata)))
            self.status = "Jumped to map position."
            self._set(it=it, av=self.res.av_grid[ia])
        elif ev.inaxes is self.ax and ev.button == 3 and self.mask_mode and ev.xdata is not None:
            um = self.res.fitter.user_masks
            hit = [m for m in um if m[0] <= ev.xdata <= m[1]]
            for m in hit:
                um.remove(m)
            if hit:
                self.status = "Unmasked %d region(s)." % len(hit)
                self.redraw()

    def run(self):
        try:
            self.plt.show()
        finally:
            for k, v in self._saved_keymaps.items():
                self.plt.rcParams[k] = v
        return self.res, self.accepted


def eyecheck(result, **kwargs):
    """Open the GUI and block until the window is closed. Returns (result, accepted)."""
    return EyeCheck(result, **kwargs).run()
