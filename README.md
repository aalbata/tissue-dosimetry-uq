# Database-Traceable Uncertainty Quantification for Layered-Tissue RF Dosimetry

Code for quantifying how the uncertainty in tabulated tissue properties propagates
to the specific absorption rate (SAR) and the steady-state temperature rise in a
layered skin/fat/muscle model, with the uncertainty tied to the IT'IS database
itself rather than to an assumed distribution.

The deterministic forward model (multilayer transmission-line SAR plus a 1D Pennes
bioheat solve) is a standard method and is not the point of this code. The point is
the uncertainty layer on top of it: conformal conductivity intervals, bounded thermal
sampling from the database min/max, Monte Carlo propagation, and Sobol sensitivity.
Everything runs from the public IT'IS database with open code. No proprietary
full-wave solver is required.

## What the code does

- `cole_cole.py` evaluates the four-term Cole-Cole dielectric model from IT'IS V5.0
  and checks it against reference tissue values (muscle and skin match to three
  significant figures).
- `dosimetry.py` solves plane-wave SAR in a layered slab by the transmission-line
  method, with an energy-conservation check.
- `bioheat.py` solves the 1D steady-state Pennes bioheat equation and returns the
  exposure-induced temperature rise.
- `uq.py` builds the split-conformal conductivity intervals and the Monte Carlo
  propagation utilities.
- `revision.py` is the main analysis. It runs the energy audit at all three
  frequencies, the Sobol first-order and total-effect indices with bootstrap
  confidence intervals (via SALib), the nominal values and central 90% intervals,
  and the repeated-split conformal coverage check. It writes `revision_results.json`.
- `figs_rev.py` regenerates the result figures (SAR and temperature depth profiles,
  interval-width summary, Sobol property and tissue panels, conformal check).
- `eqs_and_ga.py` renders the equation images and the graphical abstract.

## Reproduce

```bash
pip install -r requirements.txt
python download_data.py     # fetches the IT'IS V5.0 table from the official source
python cole_cole.py         # dielectric model check
python dosimetry.py         # SAR solver + energy check
python bioheat.py           # nominal temperature rise
python revision.py          # energy audit (3 freqs), Sobol indices + CIs, conformal coverage, bands
python figs_rev.py          # result figures (PDF)
python eqs_and_ga.py        # equation images + graphical abstract
```

## Headline results

Skin/fat/muscle slab, 50 W/m^2. Energy-balance error below 3e-4 at all three
frequencies. At 2.45 GHz, peak SAR is 6.6 W/kg nominal with a central 90% interval
of about [4.0, 8.5] W/kg, and the temperature rise spans about [0.24, 0.58] degrees C.
By Sobol total-effect indices, SAR uncertainty is governed by conductivity at all
frequencies; the temperature driver shifts from perfusion near 0.9 GHz to
conductivity near 5.8 GHz, and the critical tissue shifts from muscle to fat.
Conformal interval coverage is 0.88 to 0.89 against the 0.90 target over 50 splits.

## Data and attribution

This repository does not redistribute the IT'IS database. `download_data.py`
retrieves it directly from the IT'IS Foundation, which distributes it under its own
license and requires attribution. See https://itis.swiss/database. The database must
be acknowledged as:

> C. Baumgartner et al., "IT'IS Database for thermal and electromagnetic parameters
> of biological tissues," Version 5.0, 2025, doi:10.13099/VIP21000-05-0.

## Layout

```
cole_cole.py  dosimetry.py  bioheat.py  uq.py  revision.py  figs_rev.py  eqs_and_ga.py
download_data.py  requirements.txt  LICENSE
figures/   generated figures (PDF)
```

## Notes

- The uncertainty layer treats the dosimetric output as Y = f(theta) and would wrap
  any deterministic forward model, not just the slab used here.
- Sobol indices use a Saltelli design with bootstrap confidence intervals. The
  central interval is the 5th-to-95th percentile of the Monte Carlo output.

## Citation

If you use this code, please cite the associated paper (currently under review).
Citation metadata is in `CITATION.cff`.

## License

MIT, see `LICENSE`.
