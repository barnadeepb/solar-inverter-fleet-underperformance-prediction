# Pre-submission claim audit

Audit of `PESA_Full_Paper.docx` against the underlying results files, the
cited literature, and IEEE reference conventions. `PESA_Full_Paper_AUDITED.docx`
is a commented copy with these findings anchored to their paragraphs (Word
comments, author "Claim Audit"); this file is the standalone summary.

All must-fix and moderate findings below are already corrected in
`PESA_Full_Paper.docx` and `generate_paper.py`. Nothing here requires further
action except the two items marked "not fixed, needs your input" at the end.

## Must-fix (corrected)

**The split-comparison claim was wrong, including its direction.** The
draft claimed, in three places (abstract, an introduction contribution
bullet, and the discussion), that a random train-test split "materially
overstates accuracy" and that the chronological split "changes the reported
RMSE by roughly thirty percent." That number came from an earlier,
uncommitted session and was never reproduced against this repository's
actual code.

I ran the comparison for real (`src/split_comparison.py`, results in
`results/metrics/split_comparison.json`): under a random 80/20 split, every
model's RMSE is *higher* than under the chronological split, not lower --
the opposite of what the draft claimed. The real, verified effect is
different and, honestly, more interesting: **split choice changes which
model class wins, not the absolute error level.** Under a random split, tree
ensembles beat linear regression by roughly 12% (67.8-68.0 kW vs. 76.2 kW).
Under the chronological split, that gap nearly vanishes -- linear regression
comes within 2% of the ensembles (65.2-66.7 kW across all four). The naive
baseline itself scores slightly *better* under the chronological split
(364.67 vs. 380.60 kW), because the final ten days happen to have less
variable weather than a random 20% sample of the full window.

I rewrote the abstract, the contribution bullet, the discussion paragraph,
and added a new paragraph in the Regression Benchmark results section
presenting the corrected, fully reproducible finding with real numbers
pulled live from `split_comparison.json`.

**A specific number was misstated: "sustained over thirty-two to
thirty-four days."** The two flagged inverters are actually below the
control-chart band on 32 and 30 of the 34 study days respectively (per
`results/metrics/inverter_performance_ratios.csv`), not "32 to 34" -- 34 is
just the total window length. Fixed to read "flagged on 30 to 32 of the 34
study days," computed directly from the CSV in `generate_paper.py` rather
than typed from memory.

## Moderate (corrected)

**Reference [Vertex AI docs] pointed to a dead URL.** The original link
(`cloud.google.com/.../tabular-workflows/introduction`) 301-redirects to a
page that then 404s -- Google restructured the Vertex AI docs onto the
`docs.cloud.google.com` domain. Replaced with a URL I confirmed returns
HTTP 200 (`.../tabular-data/tabular-workflows/e2e-automl`). Worth a live
check again close to submission, since vendor docs restructure without
notice.

**The Antonanzas et al. characterization overclaimed precision I couldn't
verify.** The draft said the paper "reviews over seventy forecasting
studies." I could not reach the primary source (ScienceDirect: 403; Harvard
ADS: 405; Crossref: no abstract on file) -- what I have is a search-engine
summary describing "70 different solar irradiance prediction techniques,
53% machine-learning-based," which is *techniques*, not necessarily
*studies*, and is a secondary characterization rather than a confirmed
primary-source quote. Reworded to "a large body of forecasting techniques"
/ "a majority of the approaches reviewed" to stop short of a precise count
I can't back with the actual text. If you have institutional access to
*Solar Energy* vol. 136, a two-minute check of the real abstract could
restore a firmer number.

## Minor / informational

- **Citation order.** IEEE numbers references in order of first citation;
  the draft's mechanical check found they weren't (first-cited order was
  [2,3,4,5,17,1,6,...] against a listed [1,2,3,...]). Reordered the
  reference list and remapped every in-text `[N]` in one pass
  (`CITATION_ORDER` / `remap_citations()` in `generate_paper.py`) rather
  than hand-editing scattered numbers, to avoid a partial-renumbering bug.
- **Authors'-background page** (last sheet): Email Address and Personal
  website are blank for all four authors. Not fixed -- I don't have that
  data and won't fabricate it. Fill in before submitting; per the form's
  own note this page isn't published, only used by organizers.
- **Self-similarity, for the record, not a defect.** An n-gram check
  against this repository's own `README.md` and `paper/analysis.md` shows
  about 2% and 9.5% verbatim overlap with the paper's body text
  respectively -- expected, since `analysis.md` is explicitly the notes
  this paper was drafted from, and both live in the same public repo the
  paper cites. A checker will surface this; it's self-similarity, not
  misconduct. A separate check against the unrelated public repo
  (`open-set-solar-fault-detection`) found 0% overlap.
- **docx compatibility fix (unrelated to any specific claim, but worth
  recording):** the official PESA template uses a nonstandard
  `purl.oclc.org` XML namespace throughout instead of the standard
  `schemas.openxmlformats.org` one. Word and LibreOffice tolerate it, but
  it silently broke `python-docx` (used by this audit's own tooling) and
  would break any other strict OOXML consumer. Normalized every occurrence
  in `paper-template.docx` to the standard namespace -- purely a
  compatibility fix, with a rendered-output diff confirming no visual
  change.

## What was checked and held up

- All 17 references are individually real, correctly attributed papers,
  standards, or repositories (verified via live lookup, not memory) --
  Breiman, Chen & Guestrin, Ke et al., Lundberg & Lee, Meinshausen,
  Angelopoulos & Bates, Quinonero-Candela et al., He et al., Pedregosa et
  al., Paszke et al., the IEC standard, the Kaggle dataset, and the two
  directly comparable PV/inverter fault-detection papers all check out on
  author list, venue, and year.
- No priority or novelty overclaims exist in the draft ("first," "novel,"
  "no prior work," "not aware of any" all return zero matches) -- the
  highest-yield failure mode in this kind of audit is simply absent here
  by construction.
- Every other derived number in the paper (all of Table I and Table II, the
  cross-plant R-squared collapse, the robustness curve, the uncertainty
  coverage figure, the SHAP percentages, the latency figures) is generated
  programmatically from `results/metrics/*.json` by `generate_paper.py`,
  not typed by hand, so it is guaranteed to match its source file. I
  independently recomputed each source file's headline numbers directly
  from JSON during this audit and they match what's printed in the paper
  exactly.
- Plant 2's inverter count (22, used in the Dataset section) was verified
  against the actual data rather than assumed.
- No placeholder text, broken cross-references, or figure/table numbering
  errors remain (mechanical check, `consistency_report.json`).

## Not fixed, needs your input

1. **Corresponding author and background-page contact fields** -- both
   flagged in the earlier delivery and still open; not something an audit
   should decide for you.
2. **Francisti et al. [reference] final citation metadata** -- confirmed
   real and correctly characterized, but I could not pin down final
   journal volume/issue/pages (search results show it as a 2025
   SSRN/ScienceDirect item without complete metadata surfacing). Worth a
   direct publisher-page check before camera-ready, since preprint-to-final
   metadata commonly shifts.
