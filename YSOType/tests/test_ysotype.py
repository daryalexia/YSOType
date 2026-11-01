import csv
import os
import warnings

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pytest

from ysotype.extinction import a_lambda_over_av
from ysotype.fit import fit_spectrum
from ysotype.spt import code_to_spt, spt_to_code
from ysotype.toy import mock_spectrum, toy_grid

warnings.simplefilter("ignore")


@pytest.fixture(scope="module")
def grid():
    return toy_grid()


def test_spt_codes():
    assert spt_to_code("K7") + 1 == spt_to_code("M0")
    assert code_to_spt(spt_to_code("M5.5")) == "M5.5"
    assert code_to_spt(spt_to_code("L1")) == "L1"
    assert code_to_spt(13.6) == "M5.5"


def test_f99_reference_values():
    w = np.array([0.44, 0.55, 2.2])
    a = a_lambda_over_av(w, 3.1)
    assert abs(a[1] - 1.0) < 0.03          # V
    assert abs(a[0] - (1 + 1 / 3.1)) < 0.03  # B = V + E(B-V)
    assert 0.09 < a[2] < 0.13              # K


@pytest.mark.parametrize("teff,av,r,wr,R,mode", [
    (3200, 5.0, 0.5, (0.60, 1.00), 3000, "single"),
    (2800, 20.0, 0.0, (0.60, 1.00), 3000, "single"),
    (3000, 30.0, 1.0, (1.00, 2.45), 1200, "single"),
    (2600, 45.0, 2.0, (1.00, 2.45), 1200, "per-band"),
])
def test_injection_recovery(grid, teff, av, r, wr, R, mode):
    ref = 0.75 if wr[0] < 0.9 else 1.25
    s = mock_spectrum(grid, teff, av, r, R=R, wave_range=wr, ref=ref, snr=40)
    res = fit_spectrum(s, grid, R, veiling=mode)
    row = res.summary()
    assert abs(row["Teff"] - teff) <= 100
    assert abs(row["Av"] - av) <= 0.5
    key = "r_all" if mode == "single" else "r_J"
    assert abs(row[key] - r) <= 0.1 + 0.1 * r
    assert 0.8 < row["chi2nu"] < 1.3


def test_no_veiling_mode(grid):
    s = mock_spectrum(grid, 3100, 10.0, 0.0, wave_range=(0.6, 1.0))
    res = fit_spectrum(s, grid, 3000, veiling="none")
    assert res.k == 0 and abs(res.av - 10.0) <= 0.3


def test_search_window(grid):
    s = mock_spectrum(grid, 3100, 10.0, 0.3, wave_range=(0.6, 1.0))
    res = fit_spectrum(s, grid, 3000, initial_av=3.0, search_window=(1.0, 2.0))
    assert 1.0 <= res.av <= 5.0


def test_gui_headless(grid, tmp_path):
    from ysotype.eyecheck import EyeCheck
    s = mock_spectrum(grid, 3000, 8.0, 0.6, wave_range=(0.6, 1.0))
    e = EyeCheck(fit_spectrum(s, grid, 3000), interactive=False)

    class Ev(object):
        pass
    for k in ["right", "up", "shift+down", "]", "l", "b", "d", "n", "z", "g"]:
        ev = Ev()
        ev.key = k
        e._on_key(ev)
    e.toggle_mask()
    e._on_span(0.80, 0.81)
    e.refit()
    assert e.res.fitter.user_masks == [(0.80, 0.81)]
    out = e.save_figure(str(tmp_path / "fig.pdf"))
    assert os.path.exists(out)


def _write_fake_models(grid, d):
    """Write toy templates as BT-Settl-named ASCII (Angstrom) and .7 files."""
    for i, (t, f) in enumerate(zip(grid.teff, grid.flux)):
        wa = grid.wave * 1e4
        if i % 2:
            name = d / ("lte%03d-3.5-0.0a+0.0.BT-Settl.7" % (t / 100))
            with open(str(name), "w") as fh:
                for x, y in zip(wa, np.log10(f) + 8):
                    fh.write((" %.6f %.6e\n" % (x, y)).replace("e", "D"))
        else:
            np.savetxt(str(d / ("lte%03d-3.5-0.0a+0.0.BT-Settl.spec.txt" % (t / 100))), np.c_[wa, f])
    np.savetxt(str(d / "lte030-5.0-0.0a+0.0.BT-Settl.spec.txt"), np.c_[grid.wave * 1e4, grid.flux[0]])


def test_cli_end_to_end(grid, tmp_path, monkeypatch):
    from ysotype.cli import main
    models = tmp_path / "models"
    models.mkdir()
    small = toy_grid(teffs=np.arange(2400, 3801, 200), sampling=20000)
    _write_fake_models(small, models)
    monkeypatch.chdir(tmp_path)
    main(["grid", str(models), "-o", "g.npz", "--sampling", "20000"])
    from ysotype.templates import TemplateGrid
    g = TemplateGrid.load("g.npz")
    assert g.teff.size == small.teff.size and np.all(g.logg == 3.5)
    so = mock_spectrum(small, 3000, 6.0, 0.4, wave_range=(0.6, 1.0), name="obj1")
    sn = mock_spectrum(small, 3000, 6.0, 0.4, wave_range=(1.0, 2.45), R=1200, ref=1.25, name="obj1")
    np.savetxt("obj1_vis.txt", np.c_[so.wave * 1e3, so.flux, so.err])       # nm
    np.savetxt("obj1_nir.txt", np.c_[sn.wave * 1e4, sn.flux, sn.err])       # Angstrom
    main(["obj1_vis.txt", "--grid", "g.npz", "-R", "3000", "--name", "obj1", "--no-gui",
          "--initial-spt", "M4", "--initial-av", "5", "--figure", "obj1_vis.png"])
    main(["fit", "obj1_nir.txt", "--grid", "g.npz", "-R", "1200", "--name", "obj1", "--no-gui",
          "--veiling", "per-band", "--no-figure"])
    rows = list(csv.DictReader(open("ysotype_results.csv")))
    assert len(rows) == 2 and os.path.exists("obj1_vis.png")
    main(["combine", "ysotype_results.csv", "-o", "adopted.csv"])
    ad = list(csv.DictReader(open("adopted.csv")))
    assert ad[0]["source"] == "opt+NIR"
    assert abs(float(ad[0]["Av"]) - 6.0) < 0.5
    assert abs(spt_to_code(ad[0]["SpT"]) - spt_to_code(rows[0]["SpT"])) <= 1
