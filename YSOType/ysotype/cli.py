"""Command-line interface.

    ysotype fit SPECTRUM --grid GRID.npz -R 3000 [--initial-spt M5 --initial-av 10] ...
    ysotype grid MODEL_DIR -o GRID.npz [--logg 3.5]
    ysotype combine results.csv -o adopted.csv
    ysotype demo

``ysotype SPECTRUM ...`` (no sub-command) is the same as ``ysotype fit``.
From IPython:  %run -m ysotype fit spec.fits --grid bt_settl.npz -R 3000
"""
import argparse
import csv
import datetime
import os
import sys

import numpy as np

COLUMNS = ["name", "instrument", "R", "segments", "veiling_mode", "SpT", "SpT_code", "SpT_err",
           "Teff", "Teff_lo", "Teff_hi", "Av", "Av_lo", "Av_hi"]
for _s in ("all", "opt", "J", "H", "K"):
    COLUMNS += ["r_%s" % _s, "r_%s_lo" % _s, "r_%s_hi" % _s, "lam_ref_%s" % _s]
COLUMNS += ["chi2nu", "n_pix", "index_SpT", "initial_SpT", "initial_Av", "method", "flag",
            "comment", "user_masks", "file", "timestamp", "ysotype_version"]

SUBCOMMANDS = ("fit", "grid", "combine", "demo")


def append_row(path, row):
    new = not os.path.exists(path)
    with open(path, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS, restval="", extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow(row)


def _build_parser():
    p = argparse.ArgumentParser(prog="ysotype", description="Spectral type, extinction and veiling of young stellar objects.")
    sub = p.add_subparsers(dest="cmd")

    f = sub.add_parser("fit", help="fit one or more spectra (one at a time, with eye-check)")
    f.add_argument("spectra", nargs="+", help="FITS or ASCII spectra")
    f.add_argument("--grid", default=os.environ.get("YSOTYPE_GRID"), help="template grid .npz (or set $YSOTYPE_GRID)")
    f.add_argument("-R", "--resolving-power", type=float, required=True, help="instrument resolving power")
    f.add_argument("--instrument", default="", help="label stored with the results (e.g. XSHOOTER-VIS, GNIRS)")
    f.add_argument("--name", help="object name (default: file name)")
    f.add_argument("--initial-spt", help="starting spectral type, e.g. M5 (diagnostic / search window only)")
    f.add_argument("--initial-av", type=float, help="starting A_V (diagnostic / search window only)")
    f.add_argument("--search-window", type=float, nargs=2, metavar=("DSPT", "DAV"),
                   help="restrict the search to +-DSPT subclasses and +-DAV mag around the initial values")
    f.add_argument("--veiling", choices=("single", "per-band", "none"), default="single")
    f.add_argument("--ref-wave", type=float, help="veiling reference wavelength (um) for --veiling single")
    f.add_argument("--range", type=float, nargs=2, metavar=("WMIN", "WMAX"), help="fit range in micron")
    f.add_argument("--mask", type=float, nargs=2, action="append", metavar=("LO", "HI"), default=[],
                   help="extra region to exclude (micron); repeatable")
    f.add_argument("--rv", type=float, default=5.0, help="R_V of the Fitzpatrick (1999) law [5]")
    f.add_argument("--av-max", type=float, default=60.0)
    f.add_argument("--av-step", type=float, default=0.1)
    f.add_argument("--teff-min", type=float)
    f.add_argument("--teff-max", type=float)
    f.add_argument("--min-snr", type=float, default=3.0)
    f.add_argument("--line-velocity", type=float, default=300.0, help="half-width (km/s) of emission-line masks")
    f.add_argument("--no-telluric-mask", action="store_true")
    f.add_argument("--no-line-mask", action="store_true")
    f.add_argument("--uncorrelated", action="store_true", help="do not inflate errors for oversampled pixels")
    f.add_argument("--wave-unit", choices=("um", "nm", "A"), help="override wavelength-unit detection")
    f.add_argument("--scale", help="alternative SpT-Teff table (CSV)")
    f.add_argument("--out", default="ysotype_results.csv", help="results table (rows are appended)")
    f.add_argument("--figure", help="publication figure path (default: <name>_ysotype.pdf when accepted)")
    f.add_argument("--no-figure", action="store_true")
    f.add_argument("--no-gui", action="store_true", help="accept the automatic fit without eye-check")

    g = sub.add_parser("grid", help="build a template grid from a directory of model spectra")
    g.add_argument("directory")
    g.add_argument("-o", "--output", default="bt_settl_grid.npz")
    g.add_argument("--pattern", default="*")
    g.add_argument("--logg", type=float, default=3.5)
    g.add_argument("--teff-min", type=float, default=1200)
    g.add_argument("--teff-max", type=float, default=5400)
    g.add_argument("--wmin", type=float, default=0.50)
    g.add_argument("--wmax", type=float, default=2.60)
    g.add_argument("--sampling", type=float, default=60000.0, help="1/dln(lambda) of the stored grid")
    g.add_argument("--label", default="BT-Settl")

    c = sub.add_parser("combine", help="combine optical and NIR results per object")
    c.add_argument("results")
    c.add_argument("-o", "--output", default="ysotype_adopted.csv")
    c.add_argument("--max-dspt", type=float, default=1.0, help="allowed optical-NIR SpT difference (subclasses)")

    sub.add_parser("demo", help="open the eye-check on a toy example (no model atmospheres needed)")
    return p


def cmd_fit(a):
    import warnings
    from . import __version__
    from .fit import FitConfig, Fitter
    from .spectrum import read_spectrum
    from .templates import TemplateGrid
    if not a.grid:
        sys.exit("ysotype: --grid is required (or set $YSOTYPE_GRID). Build one with 'ysotype grid MODEL_DIR'.")
    grid = TemplateGrid.load(a.grid)
    print(grid)
    for path in a.spectra:
        spec = read_spectrum(path, wave_unit=a.wave_unit, name=a.name if len(a.spectra) == 1 else None,
                             instrument=a.instrument)
        print(spec)
        cfg = FitConfig(av_max=a.av_max, av_step=a.av_step, rv=a.rv, veiling=a.veiling, ref_wave=a.ref_wave,
                        fit_range=a.range, mask_tellurics=not a.no_telluric_mask, mask_lines=not a.no_line_mask,
                        line_velocity=a.line_velocity, extra_masks=[tuple(m) for m in a.mask],
                        min_snr=a.min_snr, correlated_errors=not a.uncorrelated,
                        initial_spt=a.initial_spt, initial_av=a.initial_av, search_window=a.search_window,
                        teff_min=a.teff_min, teff_max=a.teff_max, scale_path=a.scale)
        res = Fitter(spec, grid, a.resolving_power, cfg).run()
        _print_summary(res.summary())
        fig_path = None if a.no_figure else (a.figure or "%s_ysotype.pdf" % spec.name)
        meta = {"file": os.path.abspath(path), "ysotype_version": __version__}

        def save(r):
            row = r.summary()
            row.update(meta)
            row["user_masks"] = ";".join("%.4f-%.4f" % m for m in r.fitter.user_masks)
            row["timestamp"] = datetime.datetime.now().isoformat(timespec="seconds")
            append_row(a.out, row)
            print("-> appended %s (%s, A_V=%.2f) to %s" % (row["name"], row["SpT"], row["Av"], a.out))

        if a.no_gui:
            save(res)
            if fig_path:
                import matplotlib
                matplotlib.use("Agg")
                from .plotting import publication_figure
                publication_figure(res, fig_path)
                print("-> figure %s" % fig_path)
        else:
            from .eyecheck import eyecheck
            res, ok = eyecheck(res, on_accept=save, figure_path=fig_path)
            if not ok:
                print("-> %s not accepted; nothing written." % spec.name)


def _print_summary(row):
    rs = "  ".join("r_%s=%s" % (k[2:], v) for k, v in row.items()
                   if k.startswith("r_") and not k.endswith(("_lo", "_hi")))
    print("   best: %s (Teff=%s K)  A_V=%s [%s,%s]  %s  chi2_nu=%s  | index guess: %s" % (
        row["SpT"], row["Teff"], row["Av"], row["Av_lo"], row["Av_hi"], rs, row["chi2nu"], row["index_SpT"]))


def cmd_grid(a):
    from .templates import TemplateGrid
    g = TemplateGrid.from_directory(a.directory, pattern=a.pattern, logg=a.logg,
                                    teff_range=(a.teff_min, a.teff_max), wave_range=(a.wmin, a.wmax),
                                    sampling=a.sampling, label=a.label)
    g.save(a.output)
    print("%r\n-> %s" % (g, a.output))


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan


def cmd_combine(a):
    from .spt import code_to_spt
    rows = list(csv.DictReader(open(a.results)))
    latest = {}
    for r in rows:  # last row per (object, optical/NIR) wins
        kind = "opt" if "opt" in r["segments"].split("+") else "nir"
        latest[(r["name"], kind)] = r
    names = sorted(set(k[0] for k in latest))
    out = []
    for n in names:
        o, i = latest.get((n, "opt")), latest.get((n, "nir"))
        use = [x for x in (o, i) if x]
        codes = np.array([_f(x["SpT_code"]) for x in use])
        cerr = np.array([max(_f(x["SpT_err"]), 0.25) for x in use])
        avs = np.array([_f(x["Av"]) for x in use])
        aerr = np.array([0.5 * (_f(x["Av_hi"]) - _f(x["Av_lo"])) for x in use])
        wts = 1 / cerr ** 2
        code = np.sum(wts * codes) / np.sum(wts)
        spt_err = max(np.sqrt(1 / np.sum(wts)), 0.25)
        flag = []
        if len(use) == 2:
            d = abs(codes[0] - codes[1])
            spt_err = max(spt_err, d / 2)
            if d > a.max_dspt:
                flag.append("SpT_opt_NIR_mismatch(%.2f)" % d)
            av = avs.mean()
            av_err = np.sqrt(np.mean(aerr ** 2) + (0.5 * (avs[0] - avs[1])) ** 2)
        else:
            av, av_err = avs[0], aerr[0]
        flag += [x["flag"] for x in use if x.get("flag")]
        rec = {"name": n, "SpT": code_to_spt(code), "SpT_err": round(spt_err, 2),
               "Av": round(av, 2), "Av_err": round(av_err, 2),
               "SpT_opt": o["SpT"] if o else "", "SpT_NIR": i["SpT"] if i else "",
               "Av_opt": o["Av"] if o else "", "Av_NIR": i["Av"] if i else "",
               "source": "+".join(k for k, x in (("opt", o), ("NIR", i)) if x), "flag": ";".join(flag)}
        for s in ("all", "opt", "J", "H", "K"):
            vals = [x.get("r_%s" % s, "") for x in use if x.get("r_%s" % s, "")]
            if s == "all":
                rec["r_opt_single"] = o.get("r_all", "") if o else ""
                rec["r_NIR_single"] = i.get("r_all", "") if i else ""
            elif vals:
                rec["r_%s" % s] = vals[-1]
        out.append(rec)
    keys = []
    for r in out:
        keys += [k for k in r if k not in keys]
    with open(a.output, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, restval="")
        w.writeheader()
        w.writerows(out)
    print("%d objects -> %s" % (len(out), a.output))


def cmd_demo(a):
    from .eyecheck import eyecheck
    from .fit import fit_spectrum
    from .toy import mock_spectrum, toy_grid
    print("Toy demo: synthetic 'templates' (blackbody + fake bands), NOT model atmospheres.")
    g = toy_grid()
    s = mock_spectrum(g, 2800, 15.0, 0.8, wave_range=(1.0, 2.45), snr=30, name="demo_M6_Av15_r0.8", ref=1.25)
    res = fit_spectrum(s, g, 1200, veiling="per-band", initial_spt="M4", initial_av=10)
    _print_summary(res.summary())
    eyecheck(res, figure_path=None)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] not in SUBCOMMANDS and argv[0] not in ("-h", "--help", "--version"):
        argv.insert(0, "fit")
    if argv[:1] == ["--version"]:
        from . import __version__
        print(__version__)
        return
    a = _build_parser().parse_args(argv)
    if a.cmd is None:
        _build_parser().print_help()
        return
    {"fit": cmd_fit, "grid": cmd_grid, "combine": cmd_combine, "demo": cmd_demo}[a.cmd](a)


if __name__ == "__main__":
    main()
