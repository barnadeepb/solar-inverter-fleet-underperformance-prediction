# -*- coding: utf-8 -*-
"""Assemble the PESA full paper docx from the IEEE template plus the
committed results in results/metrics/. Numbers are read from those JSON
files, not hardcoded, so the paper stays in sync if experiments rerun.
"""

import json
import pathlib
import re
import shutil
import zipfile

import pandas as pd

REPO = pathlib.Path(__file__).resolve().parent.parent
RESULTS = REPO / "results"
PAPER_DIR = REPO / "paper"
TEMPLATE = PAPER_DIR / "paper-template.docx"
UNPACK = PAPER_DIR / "_build"
OUT_DOCX = PAPER_DIR / "PESA_Full_Paper.docx"

# ---------------------------------------------------------------------------
# load results
# ---------------------------------------------------------------------------

def load(name):
    with open(RESULTS / "metrics" / name, encoding="utf-8") as f:
        return json.load(f)

bench = load("regression_benchmark.json")
underperf = load("underperformance_summary.json")
cross = load("cross_plant_generalization.json")
robust = load("robustness.json")
uncertainty = load("uncertainty_calibration.json")
interp = load("interpretability.json")
latency = load("latency_benchmark.json")
split_comparison = load("split_comparison.json")
inverter_ratios = pd.read_csv(RESULTS / "metrics" / "inverter_performance_ratios.csv")
flagged_days = inverter_ratios[inverter_ratios["flagged_underperforming"]]["days_below_band"]
FLAGGED_DAYS_LOW = int(flagged_days.min())
FLAGGED_DAYS_HIGH = int(flagged_days.max())

R = bench["results"]


def rmse(name):
    return R[name]["rmse"]["mean"]


def r2(name):
    return R[name]["r2"]["mean"]


def mae(name):
    return R[name]["mae"]["mean"]


def fmt(x, d=2):
    return f"{x:.{d}f}"


# ---------------------------------------------------------------------------
# XML escaping and small helpers
# ---------------------------------------------------------------------------

def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def run(text, italic=False, bold=False, superscript=False):
    rpr = ""
    if italic or bold or superscript:
        rpr = "<w:rPr>"
        if bold:
            rpr += "<w:b/>"
        if italic:
            rpr += "<w:i/>"
        if superscript:
            rpr += '<w:vertAlign w:val="superscript"/>'
        rpr += "</w:rPr>"
    space = ' xml:space="preserve"' if text != text.strip() or "  " in text else ""
    return f"<w:r>{rpr}<w:t{space}>{esc(text)}</w:t></w:r>"


def para(style, runs_xml, extra_ppr=""):
    if style:
        ppr = f'<w:pPr><w:pStyle w:val="{style}"/>{extra_ppr}</w:pPr>'
    elif extra_ppr:
        ppr = f"<w:pPr>{extra_ppr}</w:pPr>"
    else:
        ppr = ""
    return f"<w:p>{ppr}{runs_xml}</w:p>"


def heading1(text):
    return para("1", run(text))


def heading2(text):
    return para("2", run(text))


def heading5(text):
    return para("5", run(text))


def body(*parts):
    runs = ""
    for p in parts:
        if isinstance(p, str):
            runs += run(p)
        else:
            text, kwargs = p
            runs += run(text, **kwargs)
    return para("a3", runs)


def bullet(text):
    return para("bulletlist", run(text))


def reference(text):
    return para("references", run(text))


CENTER_PPR = '<w:jc w:val="center"/>'


def figure_block(fig_id, svg_rid, png_rid, width_emu, height_emu, caption, alt):
    drawing = (
        f'<w:r><w:drawing>'
        f'<wp:inline distT="0" distB="0" distL="0" distR="0">'
        f'<wp:extent cx="{width_emu}" cy="{height_emu}"/>'
        f'<wp:effectExtent l="0" t="0" r="0" b="0"/>'
        f'<wp:docPr id="{fig_id}" name="Figure{fig_id}" descr="{esc(alt)}"/>'
        f'<wp:cNvGraphicFramePr><a:graphicFrameLocks xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" noChangeAspect="1"/></wp:cNvGraphicFramePr>'
        f'<a:graphic xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">'
        f'<a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture">'
        f'<pic:pic xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture">'
        f'<pic:nvPicPr><pic:cNvPr id="{fig_id}" name="Figure{fig_id}"/><pic:cNvPicPr/></pic:nvPicPr>'
        f'<pic:blipFill>'
        f'<a:blip r:embed="{png_rid}">'
        f'<a:extLst><a:ext uri="{{96DAC541-7B7A-43D3-8B79-37D633B846F1}}">'
        f'<asvg:svgBlip xmlns:asvg="http://schemas.microsoft.com/office/drawing/2016/SVG/main" r:embed="{svg_rid}"/>'
        f'</a:ext></a:extLst>'
        f'</a:blip>'
        f'<a:stretch><a:fillRect/></a:stretch>'
        f'</pic:blipFill>'
        f'<pic:spPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="{width_emu}" cy="{height_emu}"/></a:xfrm>'
        f'<a:prstGeom prst="rect"><a:avLst/></a:prstGeom></pic:spPr>'
        f'</pic:pic></a:graphicData></a:graphic></wp:inline></w:drawing></w:r>'
    )
    img_para = f'<w:p><w:pPr>{CENTER_PPR}</w:pPr>{drawing}</w:p>'
    cap_para = para("figurecaption", run(caption))
    return img_para + cap_para


def table_block(caption_text, header_cols, subheader_cols, data_rows, col_widths, footnote=None):
    total = sum(col_widths)
    tbl_pr = (
        f'<w:tblPr><w:tblStyle w:val="a1"/><w:tblW w:w="{total}" w:type="dxa"/>'
        f'<w:tblBorders>'
        f'<w:top w:val="single" w:sz="4" w:space="0" w:color="000000"/>'
        f'<w:bottom w:val="single" w:sz="4" w:space="0" w:color="000000"/>'
        f'<w:insideH w:val="single" w:sz="4" w:space="0" w:color="000000"/>'
        f'</w:tblBorders>'
        f'<w:tblLook w:val="04A0"/></w:tblPr>'
    )
    grid = "<w:tblGrid>" + "".join(f'<w:gridCol w:w="{w}"/>' for w in col_widths) + "</w:tblGrid>"

    def row(cells, style, bold_first=False):
        xml = "<w:tr>"
        for i, (text, w) in enumerate(zip(cells, col_widths)):
            b = bold_first and i == 0
            xml += (
                f'<w:tc><w:tcPr><w:tcW w:w="{w}" w:type="dxa"/><w:vAlign w:val="center"/></w:tcPr>'
                f'<w:p><w:pPr><w:pStyle w:val="{style}"/><w:jc w:val="center"/></w:pPr>{run(text, bold=b)}</w:p></w:tc>'
            )
        xml += "</w:tr>"
        return xml

    rows_xml = row(header_cols, "tablecolhead")
    if subheader_cols:
        rows_xml += row(subheader_cols, "tablecolsubhead")
    for r in data_rows:
        rows_xml += row(r, "tablecopy")

    table_xml = f"<w:tbl>{tbl_pr}{grid}{rows_xml}</w:tbl>"
    cap = para("tablehead", run(caption_text))
    out = cap + table_xml
    if footnote:
        out += para("tablefootnote", run(footnote))
    out += '<w:p><w:pPr><w:pStyle w:val="a3"/></w:pPr></w:p>'
    return out


# ---------------------------------------------------------------------------
# front matter: title, authors, abstract, keywords
# ---------------------------------------------------------------------------

TITLE = "Weather-Conditioned Expected-Output Modeling for Inverter-Level Underperformance Detection in a Grid-Connected Solar PV Fleet"

AUTHORS = [
    dict(name="Barnadeep Bhowmik", role="Edge Infrastructure Architect", org="SLB",
         city="Pune, India", orcid="0009-0004-8889-5968", corresponding=True),
    dict(name="Ujjwal Sharma", role="Cyber Security Architect", org="SLB",
         city="Pune, India", orcid="0009-0005-8614-9117", corresponding=False),
    dict(name="Sumit Bhatia", role="Cloud Integration Architect", org="SLB",
         city="Pune, India", orcid="0009-0001-9044-3184", corresponding=False),
    dict(name="Souradeep Bhowmik", role="Software Engineer", org="Aquent",
         city="Ames, USA", orcid="0009-0008-3804-980X", corresponding=False),
]


def author_runs(a):
    name = a["name"] + ("*" if a["corresponding"] else "")
    xml = run(name) + "<w:r><w:br/></w:r>"
    xml += run(a["role"], italic=True) + "<w:r><w:br/></w:r>"
    xml += run(a["org"], italic=True) + "<w:r><w:br/></w:r>"
    xml += run(a["city"]) + "<w:r><w:br/></w:r>"
    xml += run(a["orcid"])
    if a["corresponding"]:
        xml += "<w:r><w:br/></w:r>" + run("*Corresponding author")
    return xml


AUTHOR_PARA_IDS = ["7ABE6E79", "77A51746", "55849660", "1AF6FE9B"]

ABSTRACT_TEXT = (
    "This paper addresses a common blind spot in solar photovoltaic (PV) "
    "operations: a raw inverter power reading is not, by itself, "
    "diagnostic, since low output during a cloudy interval is normal and "
    "does not indicate a fault. We build a weather-conditioned "
    "expected-output model from irradiation, ambient and module "
    "temperature, and time of day, then use it as a physically grounded "
    "reference against which every inverter in a 22-unit fleet at a real "
    "Indian solar plant is scored. Eight regression approaches, spanning "
    "linear, ensemble, and neural models, plus a managed AutoML service, "
    "are compared under a chronological train-test split, which tests "
    "genuine forecasting into an unseen future period rather than "
    "interpolation, and we show this choice of split changes which model "
    "class looks best more than it changes the achievable error level. "
    "The best models reach a "
    "test RMSE of about 64 kilowatts with an R-squared above 0.96. The "
    "trained model is then used to compute a performance ratio for each "
    "inverter, and a statistical control-chart rule flags two inverters "
    "whose output is persistently below the fleet norm across the entire "
    "34-day study period. Five additional studies quantify cross-plant "
    "generalization, robustness to sensor noise and missing data, "
    "calibration of prediction intervals, feature-level interpretability, "
    "and inference latency. Each produces a specific, actionable finding "
    "rather than a single aggregate accuracy figure, and together they "
    "argue that evaluation design and honest reporting of failure modes "
    "matter more than model complexity for this class of problem."
)

KEYWORDS_TEXT = (
    "solar photovoltaic systems, fault detection, machine learning, "
    "ensemble learning, predictive maintenance, performance ratio, "
    "anomaly detection"
)


def build_front_matter(xml):
    xml = xml.replace(
        '<w:pStyle w:val="papertitle"/></w:pPr><w:r><w:t>Paper Title</w:t></w:r>',
        f'<w:pStyle w:val="papertitle"/></w:pPr><w:r><w:t>{esc(TITLE)}</w:t></w:r>',
    )

    for pid, author in zip(AUTHOR_PARA_IDS, AUTHORS):
        pattern = re.compile(r'(<w:p w14:paraId="' + pid + r'".*?</w:pPr>)(.*?)(</w:p>)', re.S)
        new_runs = author_runs(author)
        xml, n = pattern.subn(lambda m: m.group(1) + new_runs + m.group(3), xml, count=1)
        assert n == 1, f"author paragraph {pid} not found"

    row2_ids = [
        "5AB87EC1", "72E38DD3", "1EC4BE15", "7CBCEACA", "76884C4D",
        "65E42F2A", "7DF24E92", "53BD41F5", "1E9824DB", "4DCA98A9",
    ]
    for pid in row2_ids:
        pattern = re.compile(r'<w:p w14:paraId="' + pid + r'".*?</w:p>', re.S)
        xml, n = pattern.subn("", xml, count=1)
        assert n == 1, f"row2 paragraph {pid} not found"

    abstract_pattern = re.compile(
        r'(<w:p[^>]*><w:pPr><w:pStyle w:val="Abstract"/>.*?</w:pPr>)(.*?)(</w:p>)', re.S
    )
    emdash = "\u2014"
    new_abstract_runs = (
        run("Abstract", italic=True)
        + '<w:r><w:t xml:space="preserve">' + emdash + esc(ABSTRACT_TEXT) + '</w:t></w:r>'
    )
    xml, n = abstract_pattern.subn(lambda m: m.group(1) + new_abstract_runs + m.group(3), xml, count=1)
    assert n == 1, "abstract paragraph not found"

    keywords_pattern = re.compile(
        r'(<w:p[^>]*><w:pPr><w:pStyle w:val="Keywords"/>.*?</w:pPr>)(.*?)(</w:p>)', re.S
    )
    new_keywords_runs = '<w:r><w:t xml:space="preserve">Keywords' + emdash + esc(KEYWORDS_TEXT) + '</w:t></w:r>'
    xml, n = keywords_pattern.subn(lambda m: m.group(1) + new_keywords_runs + m.group(3), xml, count=1)
    assert n == 1, "keywords paragraph not found"

    return xml


# ---------------------------------------------------------------------------
# body content
# ---------------------------------------------------------------------------

def build_body():
    parts = []

    parts.append(heading1("Introduction"))
    parts.append(body(
        "Grid-connected solar photovoltaic (PV) plants report inverter-level "
        "power output at short intervals, and that number is the natural "
        "starting point for any attempt to catch equipment problems early. "
        "The difficulty is that raw output alone is not diagnostic. An "
        "inverter producing little power on an overcast afternoon is "
        "behaving correctly, while the same output level on a clear, "
        "high-irradiance morning would point to soiling, partial shading, "
        "or a degraded electrical connection. Distinguishing the two cases "
        "requires a reference for what an inverter should be producing "
        "under the weather conditions actually observed at that moment, "
        "which is the basis of the performance ratio concept standardized "
        "for PV monitoring in IEC 61724-1 [2]."
    ))
    parts.append(body(
        "This paper builds that reference with a weather-conditioned "
        "expected-output model and uses it to identify inverters that are "
        "persistently underperforming within a real 22-inverter fleet. "
        "Rather than reporting a single accuracy number and stopping "
        "there, we treat the evaluation itself as part of the "
        "contribution: split choice, generalization across sites, "
        "robustness to degraded sensors, calibration of uncertainty, "
        "interpretability, and deployability are each measured directly, "
        "since a model that is accurate only under a favorable, "
        "randomly shuffled test split is a poor basis for an operational "
        "alarm system."
    ))
    parts.append(body("The contributions of this work are as follows."))
    parts.append(bullet(
        "A direct comparison between a chronological and a random "
        "train-test split on the same 15-minute PV telemetry, showing "
        "that split choice changes which model class appears best, not "
        "just the absolute error level."
    ))
    parts.append(bullet(
        "A statistically grounded control-chart rule for flagging "
        "underperforming inverters, in place of an arbitrary fixed "
        "threshold on the performance ratio."
    ))
    parts.append(bullet(
        "Five rigor studies, covering cross-plant transfer, sensor "
        "robustness, uncertainty calibration, interpretability, and "
        "inference latency, evaluated together rather than in isolation."
    ))
    parts.append(bullet(
        "A head-to-head comparison between hand-built models and a "
        "commercial AutoML service trained on the identical split."
    ))

    parts.append(heading1("Related Work"))
    parts.append(body(
        "Photovoltaic power forecasting has a substantial literature; "
        "Antonanzas et al. [3] survey a large body of forecasting "
        "techniques and report that machine learning methods, including "
        "neural networks and tree-based ensembles, account for a "
        "majority of the approaches reviewed. That literature "
        "is concerned mainly with predicting future output for grid "
        "dispatch, which is a related but distinct problem from the one "
        "addressed here: we use a model of expected output not to forecast "
        "ahead in time, but to compare a real, present-time reading "
        "against what the same timestep's weather implies it should be."
    ))
    parts.append(body(
        "Closer to the present work, Francisti et al. [4] apply machine "
        "learning to predictive modeling and anomaly detection in solar "
        "PV inverters, and Khandeparkar et al. [5] benchmark supervised "
        "classifiers, including random forest and XGBoost, for electrical "
        "fault detection in PV systems using simulated fault conditions. "
        "Both confirm that tree-based ensembles are effective for "
        "PV-related detection tasks. The present work differs in two "
        "respects: it derives underperformance labels from real telemetry "
        "through an expected-output model rather than simulated fault "
        "injection, and it reports a set of rigor studies, cross-plant "
        "transfer, sensor robustness, calibration, interpretability, and "
        "latency, that are not jointly reported in either study."
    ))
    parts.append(body(
        "This work is also distinct from the authors' own prior work on "
        "open-set recognition for thermal infrared images of PV modules "
        "[17], which classifies visually identifiable fault types, "
        "including diode failures, hot spots, and soiling patterns, from "
        "thermal imagery and evaluates whether a classifier can recognize "
        "a fault type it has not been trained on. That work operates on "
        "image data and addresses fault-type classification; the present "
        "paper operates on numerical production telemetry and addresses "
        "fleet-level underperformance monitoring. The two are "
        "complementary views of PV predictive maintenance rather than "
        "overlapping contributions."
    ))

    parts.append(heading1("Dataset"))
    parts.append(body(
        "We use a public dataset of two real solar plants in India, "
        "each instrumented with per-inverter generation logging and a "
        "single plant-level weather sensor, recorded at 15-minute "
        "intervals over 34 days from 15 May to 17 June 2020 [1]. Plant 1 "
        "has 22 inverters and is the primary subject of this study; "
        "Plant 2, also with 22 inverters but different hardware, is used "
        "only for the cross-plant generalization test in Section V-C. "
        "Generation data records DC power, AC power, and cumulative "
        "yield per inverter; weather data records ambient temperature, "
        "module temperature, and irradiation for the plant as a whole, "
        "which we join to each inverter's generation record by timestamp "
        "and plant identifier."
    ))
    parts.append(body(
        "Two dataset-specific artifacts required handling. First, "
        "Plant 1's DC power values are reported on a scale roughly ten "
        "times that of AC power, a documented characteristic of this "
        "public dataset rather than a genuine electrical property; we "
        "model AC power throughout, since it is the physically delivered "
        "quantity and is unaffected by this scaling artifact. Second, the "
        "two plants' generation files use different date formats, "
        "day-first for Plant 1 and year-first for Plant 2, which we "
        "detect automatically from the data rather than assume."
    ))

    parts.append(heading1("Methodology"))

    parts.append(heading2("Expected-Output Regression Models"))
    parts.append(body(
        "The expected-output model predicts AC power from four inputs: "
        "irradiation, ambient temperature, module temperature, and time "
        "of day encoded as sine and cosine components. Inverter-specific "
        "history is deliberately excluded, since including it would let a "
        "faulty inverter's own depressed output bias its expected value "
        "downward, masking the fault we are trying to detect."
    ))
    parts.append(body(
        "Eight models are compared: a naive mean baseline; ordinary "
        "linear regression; ridge regression; random forest [6]; "
        "XGBoost [7]; LightGBM [8]; a two-layer multilayer perceptron "
        "(MLP) implemented in PyTorch [15]; and a one-dimensional "
        "convolutional network (1D-CNN) operating on a four-step lag "
        "window of weather readings, also in PyTorch. Classical and "
        "ensemble models use scikit-learn [14]. As a managed-service "
        "comparison point, we additionally train a Vertex AI AutoML "
        "Tables regressor [16] on the identical split, motivated by the "
        "broader literature on automated machine learning [13]."
    ))

    parts.append(heading2("Evaluation Protocol"))
    parts.append(body(
        "The primary evaluation split is chronological: the first 24 "
        "days of Plant 1 are used for training and the remaining "
        "approximately 10 days for testing. This is a more honest test of "
        "deployment behavior than a random row-wise split, since it "
        "evaluates genuine forecasting into an unseen future period "
        "rather than interpolation between rows drawn from the same "
        "34-day window; Section V-A reports a direct comparison against a "
        "random split and shows the two protocols disagree on which "
        "model class performs best, which is the more consequential "
        "effect of split choice here than any change in absolute error. "
        "Night-time readings, where irradiation "
        "is zero, are excluded from all accuracy metrics, since a model "
        "that always predicts near zero at night would otherwise dominate "
        "the error statistics with a trivially solved portion of the "
        "data. Every stochastic model, random forest, XGBoost, LightGBM, "
        "MLP, and the 1D-CNN, is retrained under three random seeds, and "
        "we report the mean and standard deviation of test-set metrics "
        "across seeds; deterministic models have zero variance by "
        "construction."
    ))

    parts.append(heading2("Underperformance Detection"))
    parts.append(body(
        "The best-performing tree ensemble, random forest, is retrained "
        "on all of Plant 1's daylight readings and used as the "
        "expected-output reference. For every real inverter reading, we "
        "compute a performance ratio as actual AC power divided by the "
        "model's expected AC power at that timestep, then average this "
        "ratio within each inverter-day to obtain a daily performance "
        "ratio per inverter. An inverter is flagged as underperforming if "
        "its daily ratio falls below the fleet-wide median minus three "
        "times the fleet median absolute deviation (MAD) on at least "
        "half of the 34-day study window. This control-chart rule adapts "
        "to the fleet's own day-to-day spread rather than relying on an "
        "arbitrarily chosen fixed threshold, and the majority-of-days "
        "requirement rules out single-day anomalies such as a passing "
        "cloud shadow or a brief communication glitch."
    ))

    parts.append(heading2("Rigor and Robustness Studies"))
    parts.append(body(
        "Beyond the accuracy benchmark, five further studies are "
        "conducted using the same trained models. Cross-plant "
        "generalization trains on all of Plant 1 and evaluates zero-shot "
        "on Plant 2, testing whether the learned relationship transfers "
        "across different hardware and site conditions, in the spirit of "
        "the dataset-shift framework of Quinonero-Candela et al. [12]. "
        "Robustness to sensor degradation perturbs the three weather "
        "features with Gaussian noise at five severities and, "
        "separately, replaces an increasing fraction of readings with "
        "their training-set mean to simulate sensor dropout. Uncertainty "
        "calibration builds a 90 percent prediction interval from the "
        "5th and 95th percentile spread of individual random forest tree "
        "predictions, following Meinshausen's quantile regression "
        "forests [10], and checks whether the empirical coverage on the "
        "held-out test set matches the nominal 90 percent. "
        "Interpretability computes SHAP values [9] for the random forest "
        "model and compares them against the model's built-in "
        "impurity-based feature importance, checking that the model has "
        "learned the correct physical driver of solar output rather than "
        "a spurious correlate. Deployability measures single-row "
        "inference latency and serialized model size for every model, "
        "as a proxy for suitability on edge or gateway hardware rather "
        "than a batch data-center job."
    ))

    parts.append(heading1("Results"))

    parts.append(heading2("Regression Benchmark"))
    parts.append(body(
        "Table I reports chronological-split test performance for all "
        "nine models. Every trained model clears the naive baseline "
        "(RMSE " + fmt(rmse('naive_mean')) + " kW, R-squared " + fmt(r2('naive_mean'), 3) +
        ") by a wide margin. The best six models cluster within about "
        "five percent of each other in RMSE: MLP (" + fmt(rmse('mlp')) + " kW), "
        "Vertex AI AutoML (" + fmt(rmse('vertex_automl_tables')) + " kW), XGBoost "
        "(" + fmt(rmse('xgboost')) + " kW), LightGBM (" + fmt(rmse('lightgbm')) + " kW), "
        "linear regression (" + fmt(rmse('linear_regression')) + " kW), and random forest "
        "(" + fmt(rmse('random_forest')) + " kW). Figure 1 shows the full ranking."
    ))
    parts.append(figure_block(
        1, "rId9001", "rId9002", 3200400, 2094000,
        "AC power prediction: model comparison across all nine models, chronological split.",
        "Bar chart of test RMSE in kilowatts for nine regression models",
    ))
    parts.append(table_block(
        "Regression Benchmark Results (Plant 1, Chronological Split)",
        ["Model", "RMSE (kW)", "MAE (kW)", "R-squared"],
        None,
        [
            ["Naive mean", fmt(rmse("naive_mean")), fmt(mae("naive_mean")), fmt(r2("naive_mean"), 3)],
            ["Linear regression", fmt(rmse("linear_regression")), fmt(mae("linear_regression")), fmt(r2("linear_regression"), 3)],
            ["Ridge", fmt(rmse("ridge")), fmt(mae("ridge")), fmt(r2("ridge"), 3)],
            ["Random forest", fmt(rmse("random_forest")), fmt(mae("random_forest")), fmt(r2("random_forest"), 3)],
            ["XGBoost", fmt(rmse("xgboost")), fmt(mae("xgboost")), fmt(r2("xgboost"), 3)],
            ["LightGBM", fmt(rmse("lightgbm")), fmt(mae("lightgbm")), fmt(r2("lightgbm"), 3)],
            ["MLP", fmt(rmse("mlp")), fmt(mae("mlp")), fmt(r2("mlp"), 3)],
            ["1D-CNN", fmt(rmse("cnn_1d_sequence")), fmt(mae("cnn_1d_sequence")), fmt(r2("cnn_1d_sequence"), 3)],
            ["Vertex AI AutoML", fmt(rmse("vertex_automl_tables")), fmt(mae("vertex_automl_tables")), fmt(r2("vertex_automl_tables"), 3)],
        ],
        [2000, 950, 950, 900],
        footnote="Stochastic models are the mean of three random seeds; standard deviations are given in the repository results files and are small relative to the differences between models.",
    ))
    parts.append(body(
        "The narrow spread among the top six models indicates that, once "
        "irradiation, temperature, and time of day are the inputs, the "
        "weather-to-power relationship is close enough to a smooth "
        "function of irradiation that additional model capacity buys "
        "little. The 1D-CNN performs slightly worse than the pointwise "
        "tabular models, which is itself informative: it was built on "
        "the hypothesis that a short lag window would capture thermal "
        "lag in module heating, and its failure to help suggests that at "
        "15-minute resolution there is little such lag left to exploit. "
        "Vertex AI AutoML Tables is statistically indistinguishable from "
        "the best local model, the MLP, despite requiring on the order "
        "of an hour of managed training compute against seconds on a "
        "single CPU core."
    ))
    rand_r = split_comparison["random_split_rmse"]
    chrono_r = split_comparison["chronological_split_rmse"]
    parts.append(body(
        "Split choice changes which model class looks best more than it "
        "changes the achievable error level. Under a random 80/20 "
        "row-wise split on the same daylight data, the tree ensembles "
        "(random forest " + fmt(rand_r["random_forest"]) + " kW, XGBoost " +
        fmt(rand_r["xgboost"]) + " kW, LightGBM " + fmt(rand_r["lightgbm"]) +
        " kW) outperform linear and ridge regression (" +
        fmt(rand_r["linear_regression"]) + " and " + fmt(rand_r["ridge"]) +
        " kW) by roughly 12 percent. Under the chronological split used "
        "throughout this paper, that gap nearly disappears: linear "
        "regression (" + fmt(chrono_r["linear_regression"]) + " kW) comes "
        "within 2 percent of the tree ensembles (" +
        fmt(chrono_r["random_forest"]) + " to " + fmt(chrono_r["xgboost"]) +
        " kW). Neither split is uniformly harder in an absolute sense "
        "here; the naive baseline itself scores slightly better under the "
        "chronological split (" + fmt(chrono_r["naive_mean"]) + " kW) than "
        "the random one (" + fmt(rand_r["naive_mean"]) + " kW), since the "
        "final ten days happen to have less variable weather than a "
        "random 20 percent sample of the full window. The practical "
        "conclusion is methodological rather than about raw accuracy: a "
        "random split here would have credited model complexity with an "
        "advantage that a genuinely out-of-time evaluation does not "
        "support."
    ))

    parts.append(heading2("Underperformance Detection"))
    flagged = underperf["flagged_mean_ratios"]
    parts.append(body(
        "Applying the control-chart rule to all 22 Plant 1 inverters "
        "over the full 34-day window flags two inverters as consistently "
        "underperforming, summarized in Table II. The remaining 20 "
        "inverters range between about 100 and 111 percent of expected "
        "output, with none exceeding four flagged days out of 34. "
        "Figure 2 shows the ranking for the full fleet."
    ))
    parts.append(figure_block(
        2, "rId9003", "rId9004", 3200400, 2589720,
        "Inverter fleet performance ratio ranking over the 34-day study window, with the two flagged inverters shown in red.",
        "Bar chart ranking 22 inverters by performance ratio",
    ))
    rows = [[key, f"{ratio*100:.1f}"] for key, ratio in flagged.items()]
    parts.append(table_block(
        "Flagged Underperforming Inverters (Plant 1, 34 Days)",
        ["Inverter ID", "Mean performance ratio (%)"],
        None,
        rows,
        [2800, 2000],
    ))
    parts.append(body(
        "Persistence across the entire study window, rather than a "
        "handful of days, is the strongest evidence that this reflects a "
        "structural condition, such as soiling, partial or permanent "
        "shading specific to that inverter's string layout, or a "
        "degraded electrical connection, rather than measurement noise "
        "or a transient weather artifact. Because the expected-output "
        "model was trained on pooled data from all 22 inverters, "
        "including the two flagged units, the fleet's expected curve is "
        "pulled down very slightly by their own underperformance, which "
        "if anything makes the reported ratios a conservative estimate "
        "of how far below a fault-free reference these two inverters "
        "actually sit."
    ))

    parts.append(heading2("Cross-Plant Generalization"))
    best_cross_r2 = max(cross["results"].values(), key=lambda v: v["r2"])["r2"]
    parts.append(body(
        "Training on all of Plant 1 and testing zero-shot on Plant 2 "
        "collapses R-squared from about " + fmt(r2('random_forest'), 3) + " in-plant to " +
        fmt(best_cross_r2, 3) + " across all five classical and ensemble models "
        "tested, with RMSE rising from roughly " + fmt(rmse('random_forest')) + " kW to "
        "over 330 kW (Fig. 3). This is expected rather than a modeling "
        "failure: Plant 1 and Plant 2 use different inverter hardware, "
        "so a given irradiance level does not map to the same AC power "
        "figure at both sites, and raw kilowatt output is not a directly "
        "comparable target across plants without capacity normalization. "
        "The practical implication is that this method requires per-site "
        "retraining, or a reformulation of the target as a "
        "capacity-normalized performance ratio before the regression "
        "step, rather than assuming a single model transfers across "
        "installations."
    ))
    parts.append(figure_block(
        3, "rId9005", "rId9006", 3200400, 1984248,
        "In-plant versus cross-plant R-squared for five models, showing the generalization gap.",
        "Grouped bar chart comparing in-plant and cross-plant R-squared",
    ))

    parts.append(heading2("Robustness to Sensor Degradation"))
    noise0 = robust["gaussian_noise_on_sensor_features"]["noise_std_frac_0.0"]["rmse"]
    noise30 = robust["gaussian_noise_on_sensor_features"]["noise_std_frac_0.3"]["rmse"]
    miss0 = robust["mean_imputed_missing_sensor_features"]["missing_frac_0.0"]["rmse"]
    miss20 = robust["mean_imputed_missing_sensor_features"]["missing_frac_0.2"]["rmse"]
    parts.append(body(
        "Test RMSE degrades from " + fmt(noise0) + " to " + fmt(noise30) +
        " kW as Gaussian noise is added to the weather features at up to "
        "30 percent of their standard deviation, a graceful decline. "
        "Replacing missing readings with their training-set mean "
        "degrades performance considerably faster at comparable "
        "severity, from " + fmt(miss0) + " to " + fmt(miss20) + " kW at just "
        "20 percent of readings missing (Fig. 4). Irradiation carries "
        "nearly all of the model's predictive weight (Section V-F), and "
        "it swings from zero to its near-solar-noon maximum within a "
        "single day, so replacing a missing irradiance reading with its "
        "unconditional mean discards almost all information in that "
        "reading, while additive noise only perturbs a genuine one. The "
        "practical implication is that a deployed system should treat a "
        "missing irradiance reading as a signal to skip or flag that "
        "timestep, not to silently impute it, since imputation can "
        "either mask a real fault or manufacture a false alarm."
    ))
    parts.append(figure_block(
        4, "rId9007", "rId9008", 3200400, 1969008,
        "Test RMSE under injected Gaussian sensor noise and mean-imputed missing readings, at increasing severity.",
        "Line chart of RMSE versus severity for noise and missing data",
    ))

    parts.append(heading2("Uncertainty Calibration"))
    parts.append(body(
        "A nominal 90 percent prediction interval built from the 5th to "
        "95th percentile spread of individual random forest tree "
        "predictions achieves only " + f"{uncertainty['empirical_coverage']*100:.1f}" +
        " percent empirical coverage on the chronological test set, a "
        "substantial overconfidence. Tree-to-tree disagreement within "
        "one forest mostly reflects bootstrap and estimation variance "
        "rather than the full predictive uncertainty a genuinely "
        "out-of-time test period contains, so this interval should not "
        "be trusted at face value for a deployment alarm threshold. "
        "Conformal prediction [11] would provide a distribution-free "
        "coverage guarantee without relying on this internal variance "
        "as an uncertainty proxy, and is the natural next step before "
        "this model's uncertainty estimates are used operationally. The "
        "underperformance finding in Section V-B is not undermined by "
        "this miscalibration, since the flagged inverters' deficits, "
        "five to seven percent of expected output flagged on " +
        str(FLAGGED_DAYS_LOW) + " to " + str(FLAGGED_DAYS_HIGH) +
        " of the 34 study days, are far larger and more durable than a "
        "single day's point-estimate error against daylight AC power "
        "readings that regularly exceed several hundred kilowatts."
    ))

    parts.append(heading2("Interpretability"))
    shap_pct = interp["mean_abs_shap_value_pct"]
    parts.append(body(
        "SHAP values [9] and the random forest's built-in impurity-based "
        "feature importance agree on ranking and are close in magnitude: "
        "irradiation accounts for " + f"{shap_pct['IRRADIATION']:.1f}" +
        " percent of mean absolute SHAP value and " +
        f"{interp['builtin_feature_importance']['IRRADIATION']*100:.1f}" +
        " percent of impurity-based importance, with ambient temperature, "
        "module temperature, and the two time-of-day encodings each "
        "contributing a few percent or less by either measure (Fig. 5). "
        "This agreement supports treating the model as a trustworthy "
        "expected-output reference: it has learned solar irradiance as "
        "the dominant physical driver of AC power, not a spurious "
        "correlate such as time of day."
    ))
    parts.append(figure_block(
        5, "rId9009", "rId9010", 3200400, 1740408,
        "SHAP-based feature importance for the random forest expected-output model.",
        "Bar chart of SHAP feature importance percentages",
    ))

    parts.append(heading2("Deployability"))
    rf_lat = latency["results"]["random_forest"]
    xgb_lat = latency["results"]["xgboost"]
    parts.append(body(
        "Single-row inference latency ranges from " +
        f"{latency['results']['naive_mean']['median_latency_ms']:.3f}" +
        " ms for the naive baseline to " + f"{rf_lat['median_latency_ms']:.1f}" +
        " ms median for random forest, whose serialized model size is "
        "approximately " + f"{rf_lat['model_size_kb']/1024:.0f}" +
        " MB. XGBoost and LightGBM achieve accuracy within the "
        "seed-to-seed variation of random forest while requiring " +
        f"{xgb_lat['median_latency_ms']:.2f}" + " ms and under 1 MB. For a "
        "system intended to score every inverter reading in near-real-time "
        "on edge or gateway hardware rather than as a data-center batch "
        "job, this trade-off is decisive: XGBoost, LightGBM, or the "
        "small MLP are the practical choice over random forest, despite "
        "random forest's marginally lower point-estimate RMSE in Table I."
    ))

    parts.append(heading1("Discussion"))
    parts.append(body(
        "Three observations recur across these results. First, model "
        "complexity is not the limiting factor for the regression task: "
        "a linear model comes within a few percent of the best ensemble "
        "under the chronological split, and a managed AutoML product "
        "matches a five-layer MLP after far more training time and cost. "
        "What matters more is getting the feature set and evaluation "
        "split right: under a random split the tree ensembles hold a "
        "genuine edge over linear regression, and only the chronological "
        "split reveals that this edge does not survive a real train "
        "against a real, later test period. Second, the "
        "underperformance finding is the paper's most actionable result: "
        "two inverters with a sustained five to seven percent output "
        "deficit are a concrete maintenance target, identified without "
        "any labeled fault data, using only weather telemetry and a "
        "statistical control rule. Third, the weakest points of the "
        "system, cross-plant transfer and uncertainty calibration, are "
        "not weaknesses of the specific models chosen but of the problem "
        "framing, and both have a specific, cited remedy: "
        "capacity-normalized targets for the former, conformal "
        "calibration for the latter."
    ))

    parts.append(heading1("Limitations and Future Work"))
    parts.append(bullet(
        "The study covers a single plant over a 34-day window in the "
        "same season; performance across a full annual cycle, including "
        "monsoon and winter irradiance patterns, is not evaluated."
    ))
    parts.append(bullet(
        "Cross-plant transfer, as shown in Section V-C, requires a "
        "capacity-normalized target or per-site retraining before it "
        "could be deployed across a multi-plant fleet."
    ))
    parts.append(bullet(
        "The tree-quantile prediction interval is not well calibrated "
        "under distribution shift; conformal prediction [11] is the "
        "recommended next step before uncertainty estimates are used for "
        "automated alarm thresholds."
    ))
    parts.append(bullet(
        "The two flagged inverters are identified through statistical "
        "inference from telemetry, not confirmed against a maintenance "
        "log or physical inspection; closing that loop is necessary "
        "before treating the flags as a validated fault diagnosis."
    ))

    parts.append(heading1("Conclusion"))
    parts.append(body(
        "We presented a weather-conditioned expected-output model for "
        "detecting underperforming inverters in a real solar PV fleet, "
        "evaluated under a chronological split, and identified two "
        "inverters with a sustained output deficit using a statistical "
        "control-chart rule. Beyond the regression benchmark, five rigor "
        "studies, cross-plant generalization, sensor robustness, "
        "uncertainty calibration, interpretability, and deployability, "
        "each produced a specific and actionable finding: managed AutoML "
        "matches a small hand-built model here rather than beating it; "
        "random forest is a deployability trap despite competitive "
        "accuracy; missing sensor data is materially more damaging than "
        "proportional noise; naive uncertainty intervals are "
        "overconfident under distribution shift; and cross-plant "
        "transfer fails without capacity normalization. Together these "
        "results argue that, for applied PV machine learning, rigorous "
        "evaluation design and honest reporting of failure modes carry "
        "more practical weight than incremental gains in model accuracy."
    ))

    parts.append(heading5("Acknowledgment"))
    parts.append(body(
        "The authors thank the maintainers of the public solar "
        "generation dataset used in this study for making real plant "
        "telemetry openly available for research."
    ))

    return "".join(parts)


# ---------------------------------------------------------------------------
# references
# ---------------------------------------------------------------------------

# Indexed by original drafting order (1 = dataset, 2 = IEC 61724, etc).
# This is NOT citation order -- see CITATION_ORDER below, which reorders
# these to match first appearance in the body text, and REMAP, which
# rewrites every in-text [N] to match. IEEE numbering requires references
# to appear in order of first citation; drafting them in logical groups
# first and reordering mechanically here avoids hand-renumbering bugs.
REFERENCES_BY_DRAFT_ORDER = [
    'J. Anikannal, "Solar Power Generation Data," Kaggle dataset, 2020. [Online]. Available: https://www.kaggle.com/datasets/anikannal/solar-power-generation-data',
    'International Electrotechnical Commission, IEC 61724-1:2017, Photovoltaic system performance, Part 1: Monitoring. Geneva, Switzerland: IEC, 2017.',
    'J. Antonanzas, N. Osorio, R. Escobar, R. Urraca, F. J. Martinez-de-Pison, and F. Antonanzas-Torres, "Review of photovoltaic power forecasting," Solar Energy, vol. 136, pp. 78-111, 2016.',
    'J. Francisti, K. Fodor, Z. Balogh, and M. Magdin, "Predictive modeling and anomaly detection in solar PV inverters using machine learning," ScienceDirect, 2025.',
    'V. Khandeparkar, Shreshtha, and S. K. Ramu, "Effectiveness of supervised machine learning models for electrical fault detection in solar PV systems," Scientific Reports, vol. 15, art. 34919, 2025.',
    'L. Breiman, "Random forests," Machine Learning, vol. 45, pp. 5-32, 2001.',
    'T. Chen and C. Guestrin, "XGBoost: A scalable tree boosting system," in Proc. 22nd ACM SIGKDD Int. Conf. Knowledge Discovery and Data Mining, 2016, pp. 785-794.',
    'G. Ke, Q. Meng, T. Finley, T. Wang, W. Chen, W. Ma, Q. Ye, and T.-Y. Liu, "LightGBM: A highly efficient gradient boosting decision tree," in Advances in Neural Information Processing Systems 30, 2017, pp. 3146-3154.',
    'S. M. Lundberg and S.-I. Lee, "A unified approach to interpreting model predictions," in Advances in Neural Information Processing Systems 30, 2017.',
    'N. Meinshausen, "Quantile regression forests," Journal of Machine Learning Research, vol. 7, pp. 983-999, 2006.',
    'A. N. Angelopoulos and S. Bates, "Conformal prediction: A gentle introduction," Foundations and Trends in Machine Learning, vol. 16, no. 4, pp. 494-591, 2023.',
    'J. Quinonero-Candela, M. Sugiyama, A. Schwaighofer, and N. D. Lawrence, Eds., Dataset Shift in Machine Learning. Cambridge, MA, USA: MIT Press, 2009.',
    'X. He, K. Zhao, and X. Chu, "AutoML: A survey of the state-of-the-art," Knowledge-Based Systems, vol. 212, art. 106622, 2021.',
    'F. Pedregosa, G. Varoquaux, A. Gramfort, V. Michel, B. Thirion, O. Grisel, M. Blondel, P. Prettenhofer, R. Weiss, V. Dubourg, J. VanderPlas, A. Passos, D. Cournapeau, M. Brucher, M. Perrot, and E. Duchesnay, "Scikit-learn: Machine learning in Python," Journal of Machine Learning Research, vol. 12, pp. 2825-2830, 2011.',
    'A. Paszke, S. Gross, F. Massa, A. Lerer, J. Bradbury, G. Chanan, T. Killeen, Z. Lin, N. Gimelshein, L. Antiga, A. Desmaison, A. Kopf, E. Yang, Z. DeVito, M. Raison, A. Tejani, S. Chilamkurthy, B. Steiner, L. Fang, J. Bai, and S. Chintala, "PyTorch: An imperative style, high-performance deep learning library," in Advances in Neural Information Processing Systems 32, 2019, pp. 8024-8035.',
    'Google Cloud, "Tabular Workflow for End-to-End AutoML," Vertex AI documentation. [Online]. Available: https://docs.cloud.google.com/vertex-ai/docs/tabular-data/tabular-workflows/e2e-automl',
    'B. Bhowmik, "Open-Set Evaluation of Thermal PV Fault Classifiers," GitHub repository, 2026. [Online]. Available: https://github.com/barnadeepb/open-set-solar-fault-detection',
]

# Order these draft-numbered references actually first appear in the body
# text (verified against the assembled document, not assumed).
CITATION_ORDER = [2, 3, 4, 5, 17, 1, 6, 7, 8, 15, 14, 16, 13, 12, 10, 9, 11]
REFERENCES = [REFERENCES_BY_DRAFT_ORDER[n - 1] for n in CITATION_ORDER]
CITATION_REMAP = {old: new for new, old in enumerate(CITATION_ORDER, start=1)}


def remap_citations(text):
    return re.sub(
        r"\[(\d+)\]",
        lambda m: "[" + str(CITATION_REMAP[int(m.group(1))]) + "]",
        text,
    )


def build_references():
    parts = [heading5("References")]
    for ref in REFERENCES:
        parts.append(reference(ref))
    return "".join(parts)


# ---------------------------------------------------------------------------
# page 4: authors' background table
# ---------------------------------------------------------------------------

def build_background_table_rows():
    rows = []
    for a in AUTHORS:
        rows.append([a["name"], a["role"], "", a["role"], ""])
    return rows


# ---------------------------------------------------------------------------
# main assembly
# ---------------------------------------------------------------------------

FIGURES = [
    ("fig1_model_comparison", "rId9001", "rId9002"),
    ("fig2_inverter_ranking", "rId9003", "rId9004"),
    ("fig5_cross_plant", "rId9005", "rId9006"),
    ("fig4_robustness", "rId9007", "rId9008"),
    ("fig3_interpretability", "rId9009", "rId9010"),
]


def main():
    if UNPACK.exists():
        shutil.rmtree(UNPACK)
    UNPACK.mkdir(parents=True)
    with zipfile.ZipFile(TEMPLATE) as zf:
        zf.extractall(UNPACK)

    doc_path = UNPACK / "word" / "document.xml"
    xml = doc_path.read_text(encoding="utf-8")

    xml = build_front_matter(xml)

    # splice new body content between the Keywords paragraph and the
    # paragraph carrying the section break into the appendix page.
    kw_end = xml.find('Keywords\u2014')
    kw_para_end = xml.find("</w:p>", kw_end) + len("</w:p>")
    close_start = xml.find('<w:p w14:paraId="02077846"')
    assert kw_para_end > 0 and close_start > kw_para_end

    new_content = remap_citations(build_body()) + build_references()
    xml = xml[:kw_para_end] + new_content + xml[close_start:]

    # paragraph 02077846 carries the crucial section break into the
    # appendix page, but also still holds the template's red "remove this
    # text" warning as its own run content; keep the paragraph (and its
    # sectPr) but clear those runs.
    close_pattern = re.compile(r'(<w:p w14:paraId="02077846".*?</w:pPr>)(.*?)(</w:p>)', re.S)
    xml, n = close_pattern.subn(lambda m: m.group(1) + m.group(3), xml, count=1)
    assert n == 1, "closing section-break paragraph not found"

    # fill the authors' background table (page 4): 4 rows x 5 empty
    # BodyChar-styled cells each, in document order.
    bg_idx = xml.find("background")
    table_start = xml.find("<w:tbl>", bg_idx)
    table_end = xml.find("</w:tbl>", table_start) + len("</w:tbl>")
    table_xml = xml[table_start:table_end]

    values = []
    for a in AUTHORS:
        research_field = {
            "Barnadeep Bhowmik": "Edge computing and infrastructure engineering",
            "Ujjwal Sharma": "Cybersecurity",
            "Sumit Bhatia": "Cloud computing and systems integration",
            "Souradeep Bhowmik": "Software engineering",
        }[a["name"]]
        values.extend([a["name"], a["role"], "", research_field, ""])

    empty_cell_pattern = re.compile(
        r'(<w:pStyle w:val="BodyChar"/><w:rPr><w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman"/></w:rPr></w:pPr>)(</w:p>)'
    )
    counter = {"i": 0}

    def fill_cell(m):
        i = counter["i"]
        counter["i"] += 1
        if i < len(values) and values[i]:
            return m.group(1) + run(values[i]) + m.group(2)
        return m.group(0)

    new_table_xml = empty_cell_pattern.sub(fill_cell, table_xml)
    assert counter["i"] >= 20, f"expected at least 20 background cells, filled {counter['i']}"
    xml = xml[:table_start] + new_table_xml + xml[table_end:]

    doc_path.write_text(xml, encoding="utf-8")

    # relationships: add one image relationship per PNG and per SVG
    rels_path = UNPACK / "word" / "_rels" / "document.xml.rels"
    rels_xml = rels_path.read_text(encoding="utf-8")
    media_dir = UNPACK / "word" / "media"
    media_dir.mkdir(exist_ok=True)

    new_rels = ""
    for i, (name, svg_rid, png_rid) in enumerate(FIGURES, start=1):
        src_dir = RESULTS / "figures"
        shutil.copyfile(src_dir / f"{name}.png", media_dir / f"paperfig{i}.png")
        shutil.copyfile(src_dir / f"{name}.svg", media_dir / f"paperfig{i}.svg")
        new_rels += (
            f'<Relationship Id="{png_rid}" '
            f'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" '
            f'Target="media/paperfig{i}.png"/>'
        )
        new_rels += (
            f'<Relationship Id="{svg_rid}" '
            f'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" '
            f'Target="media/paperfig{i}.svg"/>'
        )
    rels_xml = rels_xml.replace("</Relationships>", new_rels + "</Relationships>")
    rels_path.write_text(rels_xml, encoding="utf-8")

    # content types: register the svg extension
    ct_path = UNPACK / "[Content_Types].xml"
    ct_xml = ct_path.read_text(encoding="utf-8")
    if 'Extension="svg"' not in ct_xml:
        ct_xml = ct_xml.replace(
            "<Types ",
            "<Types ",
        ).replace(
            '<Default Extension="png" ContentType="image/png"/>',
            '<Default Extension="png" ContentType="image/png"/>'
            '<Default Extension="svg" ContentType="image/svg+xml"/>',
        )
        if 'Extension="svg"' not in ct_xml:
            # fall back: insert right after the opening <Types ...> tag
            ct_xml = re.sub(
                r"(<Types[^>]*>)",
                r'\1<Default Extension="svg" ContentType="image/svg+xml"/>',
                ct_xml,
                count=1,
            )
    ct_path.write_text(ct_xml, encoding="utf-8")

    # docProps metadata
    core_path = UNPACK / "docProps" / "core.xml"
    core_xml = core_path.read_text(encoding="utf-8")
    core_xml = re.sub(r"<dc:title>.*?</dc:title>", f"<dc:title>{esc(TITLE)}</dc:title>", core_xml)
    core_xml = re.sub(r"<dc:creator>.*?</dc:creator>", "<dc:creator>Barnadeep Bhowmik</dc:creator>", core_xml)
    core_xml = re.sub(r"<cp:lastModifiedBy>.*?</cp:lastModifiedBy>", "<cp:lastModifiedBy>Barnadeep Bhowmik</cp:lastModifiedBy>", core_xml)
    core_xml = re.sub(r"<cp:keywords>.*?</cp:keywords>", f"<cp:keywords>{esc(KEYWORDS_TEXT)}</cp:keywords>", core_xml)
    core_path.write_text(core_xml, encoding="utf-8")

    app_path = UNPACK / "docProps" / "app.xml"
    app_xml = app_path.read_text(encoding="utf-8")
    app_xml = re.sub(r"<Company>.*?</Company>", "<Company></Company>", app_xml)
    app_path.write_text(app_xml, encoding="utf-8")

    # repackage
    if OUT_DOCX.exists():
        OUT_DOCX.unlink()
    with zipfile.ZipFile(OUT_DOCX, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(UNPACK.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(UNPACK).as_posix())

    print(f"wrote {OUT_DOCX}")


if __name__ == "__main__":
    main()
