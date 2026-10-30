# OCVR — Olds College Variable-Rate economics

OCVR answers one question for an on-farm trial: **would variable rate by NDVI zone have paid, compared with a flat
rate?** It puts together what the OFPE work already produces — the cleaned yield map (OCCD), the trial design, the
as-applied maps, a multi-year NDVI zone map (fetched by the app, or the grower's own zone map) and the grain, seed and
urea prices — and returns the response to the rate in each zone, the zone and flat optima, the value of VR against the
grower's status quo and against the best flat rate, how sure that value is, a prescription shapefile and a PDF report.

It follows the DIFM logic (plots, trimmed transitions, response, optimum, partial budget) but uses the **multi-year
NDVI as the variable that splits the field**, either as zones or as a continuous covariate.

Owner: Gilberto Siebert Filho, Olds College Smart Farm (OFPE). Version 1.1.0 (2026-10-29).
Sister apps: **OCCD** (`C:\Olds College Projects\App limpeza DIFM`, yield cleaning; run it first) and the **NDVI Map
Viewer** (`GIT\NDVIMAP`, whose Copernicus client and saved credentials OCVR reuses).

## Start

On a new computer: unzip `OCVR_portable.zip` anywhere (e.g. `Documents\OCVR`), install Python 3.12+ from python.org
(tick "Add python.exe to PATH"), double-click `setup.bat` once (it makes the `.venv` environment, 2–5 minutes), then
`run_app.bat`. The saved projects open from the sidebar with all their input files (the package copies inputs that
lived outside a project into its `inputs\external\`, and a moved project finds its own files in the new folder). To
fetch new NDVI, open `NDVIMAP\run.bat` once and save the Copernicus client id and secret in panel 1 (the credentials
are not in the package). Make a new package with `python tools\make_portable.py <empty folder> --zip OCVR_portable.zip`.

Double-click `run_app.bat`, or from this folder:

```
python -m streamlit run app\app.py
```

Packages: `python -m pip install -r requirements.txt` (Python 3.12+, tested with Python 3.14 and Streamlit 1.61.1).
The same work runs from the command line from a saved project:

```
python -m engine.run workspace\<project>\project.json [--no-fetch] [--B 100] [--R 30]
```

(`--B` / `--R` with fewer resamples write to `<project>\quick_B<B>_R<R>`, so the saved project is not replaced by a quick run.)

Tests: `python -m pytest -q` (synthetic trials with a known answer; the end-to-end app test needs a saved trial project and is skipped without one).

## The six steps

| Step | What you give | What comes back |
|---|---|---|
| 1 Trial | name, crop, year; the **trial design** (plots + border/headland polygon), the **cleaned yield points**, optionally the field boundary; **what was tested** (Seed, Urea, Seed and urea, or Other inputs for any other input), with the grower's **status quo** rate of each; the as-applied map of each input, and the rates of each input after the first: a second design file, a column of the main design, or **"(from the as-applied map)"** when no design holds them | plots, rates of every input and where they came from, plots per rate, field area, one map per input |
| 2 NDVI & zones | one of: *fetch* (Sentinel-2 L2A, 2017 on), NDVI GeoTIFFs you have, a ready multi-year NDVI map, or **your zone map**; the **crop history** (which seasons were the same crop) | seasons used, temporal mean and CV maps, zones with field acres |
| 3 Prices | crop sale price ($/bu or $/t), purchase price of each input ($/lb of product; N priced per lb N when the rates are N), VR cost $/ac, the ± % of the price grid | price per rate unit |
| 4 Join | trims and containment rules (defaults below) | observations kept and why the others were not, analysis units (plot × zone) |
| 5 Economics | response form, bootstrap and cross-fit sizes, minimum worthwhile gain | verdict, zone and flat optima, gains, tests, figures |
| 6 Report | — | PDF, `zones_rx.zip` (zones with the zone rate and the recommended flat rate of each input; shapefile field names are cut to 10 characters, so `zones_rx_fields.csv` in the zip says which field is which), units (GeoPackage, CSV), `results.json` |

Every project lives in `workspace\<stamp>_<name>\` (`inputs\` uploads, `ndvi_images\` downloads, `output\` results,
`project.json` the record); **Saved projects** in the sidebar opens it again.

## How it decides

1. **Zones.** For each past season (the trial year left out) the app takes the per-pixel **median of the three clearest
   Sentinel-2 dates with the highest field-mean NDVI** in the window (default 15 Jun – 15 Aug), normalises it to the
   field mean, and averages the seasons (Blackmore 2000; Lenoir et al. 2024). Seasons of the trial's crop are used when
   at least three are known, then the crop group, then all seasons (and it says which). The temporal mean is smoothed
   (3 × 3), cut into **tertiles** (k-means as an option) and patches under 0.5 ha are merged.
2. **Two inputs on one set of plots.** A seed x urea trial has two designs. The urea rates come from a second design
   file, from another column of the main design, or, when the drill logged them but no design file holds them,
   **from the urea as-applied map**: each plot takes the median rate of the points inside its trimmed
   interior, the plot medians are grouped into the levels the drill delivered, and the level is the plot's design
   rate. The number of levels is the 1-D optimal partition with the best separation score (smallest gap between two
   levels / largest spacing inside a level); it must reach 2, no other number of levels may come close, and the
   chosen levels must pass two shape tests (no level with a jump inside it far above its other spacings, no level
   much wider than the others: the signs of two design levels read as one). With fewer than 24 rated plots the
   score must reach 4 (at 8 plots, falling to 2.5 at 23: what a gradient reaches by chance on so few plots), and
   such a small design is always flagged. Otherwise the app stops and asks for a
   design file (a gradient prescription, levels that run into each other, a small design). The reading is reliable
   when the design steps are roughly equal or the plot-to-plot drift is well under half the smallest step; on the
   sweeps of `tests/test_levels.py` (unequal steps 90/100/110/130 and 100/120/200/300 with drift, gradients on 8–20
   plots, Trial-22-like maps, exactly logged check strips) it returned no wrong number of levels on designs of 20
   or more plots per level, at the cost of refusing about 1 clean map in 150; on 5-plot-per-level designs it reads
   90-94 % and refuses the rest, and a gradient on 8-20 plots slips through as levels in 2-4 % of cases, always
   with the small-design warning. A score under 3, a level that may hide a finer step, and a design under 24 plots
   are flagged in the app and the PDF. Plots with fewer than 5 points, with mixed rates (p10 and p90 nearest two different levels, or a spread
   over half a step, judged above three times the map's own point noise) or off every level get no design rate;
   they still enter the response with their applied rate (when the rate in the models is the applied one). Each
   input is analysed with the other as a control.
3. **Join.** Each yield record becomes a rectangle (swath × distance along the heading, corrected to grid north). It is
   kept when ≥ 95 % of it lies inside **one** plot after trimming **10 m** at each plot end (DIFM's ~10 m transitional
   buffers, Bullock et al. 2019; the footprint overlay of dos Santos & Miguez 2025; the yield-monitor delay itself, 10–23 s
   in Chung, Sudduth & Drummond 2002, is corrected upstream by the cleaning app, so the trim covers the applicator's rate
   change and the residual delay error); the trim is measured from every end edge, so a wedge or trapezoid plot
   clipped to a boundary loses its full 10 m along the whole width. Each plot is split by zone into analysis units; the
   applied rate of a unit is the median of its as-applied points. Units need ≥ 10 records and ≥ 150 m². The swath and
   distance units (ft or m) are chosen in step 1 (the column name only sets the default), and the join checks the
   swath against the spacing of the yield passes and the as-applied rates against the design rates (a metric export
   read as feet, a map in lb of product under a price per lb N) and warns in the app and the PDF.
4. **Response.** Quadratic response to the rate (quadratic-plateau always reported as a sensitivity), as one curve, one
   curve per zone, and a curve that changes with the continuous NDVI. Weighted by area, with combine shares, a trend
   surface and the other tested input as controls; standard errors clustered by plot. A zone with fewer than 20 units or
   3 design rates is merged into its neighbour.
5. **Economics.** Partial net return = crop price × yield − input price × rate, on the field's zone areas, with optima
   inside the trial range. VR is compared with the **status quo** and with the **best flat rate**.
6. **Verdict.** In-sample, zone rates can never lose to one flat rate, so the evidence is the **cross-fit** (rates chosen
   on half the plots, judged on the other half, 30 repeats; sample splitting as in Chernozhukov et al. 2018 and Athey &
   Imbens 2016, because the in-sample value of VR is biased upward by construction, Mieno, Li & Bullock 2024) and the
   **zone × rate test** (a cluster-robust Wald test of equal rate coefficients across zones, judged against F(h, plots − 1);
   Cameron & Miller 2015).
   *VR supported* needs a mean cross-fitted gain (per-repeat average of the two folds) ≥ the minimum worthwhile gain
   (default $3/ac, after the VR cost), positive in ≥ 70 % of the repeats, and a zone × rate p < 0.10; *weak* when the
   mean gain reaches min(1, minimum / 3) $/ac with ≥ 60 % of repeats positive and either test holds; otherwise *flat
   rate*, with the best flat rate from the single-curve model. The same gate (mean ≥ minimum, ≥ 70 % positive, with
   the in-sample gain on the field's zone acres positive) decides whether the best flat rate replaces the status quo;
   a **higher** rate must also show a field response to the rate (pooled test p < 0.10), because it has to buy its
   extra yield, while a **lower** rate saves input with certainty and is judged on its gains alone (the report then
   says how much of the gain is the input saving). A cross-fitted gain within $1/ac of the minimum is flagged
   borderline. A plot bootstrap gives the intervals of the optima and gains, and a price grid (± 30 % crop and input
   price) shows how the answer moves with prices (Pannell 2006; Pannell, Gandorfer & Weersink 2019). A lower rate advised
   without a yield response carries the caveat that under-application costs more than over-application when the true
   response is steeper than estimated (Babcock 1992; Rajsic & Weersink 2008): repeat the trial before cutting further.
7. **Realised trial.** On the analysed units, the net return of the trial as applied is compared with the status quo, the
   best flat rate and the zone rates (each unit keeps its own residual). With two inputs the cost of the trial to the
   grower (the order of the DIFM compensation) is computed once, with every input at its status quo at once; each
   input's own table holds the other input at its trial rates.
8. **Units.** Every rate inside the engine, the PDF and the prescription is lb/ac in the input's basis (lb of product,
   or lb of N for an input priced per lb N). A design or map in kg/ha is converted on loading (× 0.8922); an
   as-applied map that logs lb of product for an input priced per lb N is converted with the product's nutrient
   fraction (step 1 asks what the map logs). Yield must be dry bu/ac.

The literature behind each rule, checked paper by paper (every citation read and judged, every DOI resolved on
2026-10-29), is in `docs/literature/OCVR_method_and_references.pdf` (and `.md`; Portuguese version `OCVR_metodo_e_referencias.pdf`); `docs/LITERATURE.md` keeps the
rule-to-reference table and the verified reference list.

## Validation (2026-10-29)

**Synthetic trials with a known answer** (`tests/synth.py`: 72 plots, 5 seed rates, two NDVI zones; true zone optima
87.8 and 167.8 lb/ac at C$9.00/bu and C$0.22/lb, true best flat ≈ 128). Six trials where the zones really respond
differently and six where they only differ in yield level:

| Case | Verdict | Zone optima found | Best flat found | Cross-fitted VR gain |
|---|---|---|---|---|
| Zones respond differently (6 seeds) | VR in 6 of 6 | 80–103 and 161–183 | 124–133 | +10.8 to +16.5 $/ac |
| Same response (6 seeds) | flat in 6 of 6 | — | 84–94 (true 88) | −0.4 to +0.7 $/ac |

Two rules were added because these controls failed without them: the minimum worthwhile gain (a chance rate imbalance
between zones gave p = 0.015 and +$2/ac with no real difference), and a yield level per zone in the single-curve model
(without it the best flat rate came out at 80 instead of 128).

Results on real trials stay out of this repository: they belong to the growers. It holds the code and the
documentation only; project data lives in `workspace\`, which git ignores.

The app itself is tested end to end on a saved trial project (`tests/test_app_flow.py`: open the project, every step's
button, PDF and prescription written), and so is its "What was tested?" selector. The decision rules of item 6 are tested
directly on minimal results (`test_verdict_rules`), the level reading on sweeps (`tests/test_levels.py`). 36 tests,
about 4 minutes.

**Code review.** The engine and the app went through eight rounds of a multi-agent review (nine to four reviewers, two
independent skeptics per finding) and fix rounds with checkers: 48, 30, 23, 13, 15, 17, 8 and 26 confirmed findings.
The eighth round (2026-10-28/29, nine lenses, 76 agents) found one crash (combine names with spaces in the model
formula), one inverted rule (a cheaper flat rate refused for lack of a yield response), the plateau held flat below
the trial range, the level reading accepting merged levels on unequal steps, units and bases that were labels only,
and report inconsistencies (bootstrap medians printed as estimates, a static Methods page, two contradictory
compensation sentences); all were fixed and checked. The tests, the 12 simulations and the three trials were rerun
after every round.

## Limits and things to know

- **The crop history matters.** Without it the app uses every clear season and says so; zones from canola years and
  cereal years mixed are a weaker productivity ranking. Fill the table in step 2.
- **Peak NDVI saturates** in dense cereal canopies (CV of the normalised NDVI is often 3–5 %). The zones then rank
  yield level well but carry little information about the response; that is what the zone × rate test checks.
- **Whole-field economics need the field boundary.** Without it the zone areas are those of the trial area.
- **One season is ex post.** The optimum of one year is not next year's recommendation (Rajsic & Weersink 2008); repeat
  the trial before acting on zone rates.
- Sentinel-2 L2A starts in 2017: at most nine seasons before 2026.
- **Optima at the edge of the trial range** are corners, not measured optima: the response was not tested
  below the lowest rate. The continuous-NDVI prescription is reported as a diagnostic only when most of its bins are
  corners.
- A **grower zone map** must cover the field: cells outside every polygon are given the nearest zone and the share is
  printed; a map leaving more than 30 % of the field unclassified is refused.
- Two inputs on one design are often correlated; each input is analysed with the
  other as a control, and the report shows both.
