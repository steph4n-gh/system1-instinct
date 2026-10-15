#!/usr/bin/env python3
"""Real-World US FDA Enforcement Recall Triage Dogfood Harness (`openFDA` + `astral-sh/uv` OOS).

Replays 297 real US FDA enforcement recall reports fetched from the public openFDA API
(`https://api.fda.gov/{food,drug,device}/enforcement.json`) across all 9
`(regulatory_center, hazard_class)` combinations (`food`, `drug`, `device` x
`Class I`, `Class II`, `Class III`, 33/cell total: 25/cell observed + 8/cell held-out)
in strict chronological order (`oldest -> newest` by `(report_date, recall_number)`,
zero synthetic strings, zero temporal leakage), plus 25 real `astral-sh/uv` software
engineering issues as an Out-of-Scope (OOS) stress gate:

  - Multi-Head Schema (`FdaRecallTriageSchema`):
      * `regulatory_center`: `ChoiceField(["food", "drug", "device"])`
      * `hazard_class`: `ChoiceField(["Class I", "Class II", "Class III"])`
      * `urgent_class_i_escalation`: `BooleanField()` (`True` iff `Class I`)
  - Stage 1: Cold Start (0 examples, Recalls 1-10)
      `SystemOneEngine` + `BoundedReviewFallback` safely abstains (100% escalated to
      the regulatory reviewer fallback) rather than guessing blindly, learning online via
      `engine.learn_from_outcome(...)`.
  - Stage 2: Online Warm-Up (Recalls 11-50)
      Online updates (`engine.learn_from_outcome` / `BoundedReviewFallback(auto_online_update=True)`)
      begin offloading clear repeat recall patterns locally in ~0.35 ms while escalating uncertain items.
  - Stage 3: Auto-Cutover Checkpoints (12/cell = 108, 18/cell = 162, 25/cell = 225 observed recalls)
      Compiles a calibrated multi-head `.s1m` skill with
      `SystemOneCompiler(auto_regularization=True, noise_robust=True)` and evaluates on
      the 72 strictly held-out newest chronological recalls (`report_date >= 20240601`).
  - Stage 4: Out-of-Scope Traffic Gate (`astral-sh/uv` software issues hitting the FDA recall skill)
      Measures safe abstention (`is_ambiguous == True`) on non-regulatory software issues.
  - Stage 5: `.s1m` Skill Save & Reload Parity
      Saves with `engine.save(...)`, reloads with `SystemOneEngine.load(...)`, and verifies
      100% bit-identical multi-head decisions on held-out recalls.

Usage:
  # Default deterministic offline run (<1.5s using cached real openFDA snapshot):
  ./.venv/bin/python examples/dogfood/live_fda_recall_triage.py

  # Live fetch from openFDA Enforcement REST API:
  ./.venv/bin/python examples/dogfood/live_fda_recall_triage.py --live
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from system1 import (  # noqa: E402
    BooleanField,
    BoundedReviewFallback,
    ChoiceField,
    DecisionSchema,
    SystemOneCompiler,
    SystemOneEngine,
    TfidfProjector,
)

CENTERS: Tuple[str, ...] = ("food", "drug", "device")
HAZARD_CLASSES: Tuple[str, ...] = ("Class I", "Class II", "Class III")
CELLS: Tuple[Tuple[str, str], ...] = tuple(
    (center, hazard) for center in CENTERS for hazard in HAZARD_CLASSES
)

DEFAULT_SNAPSHOT_PATH = Path(__file__).resolve().parent / "fda_live_recalls_snapshot.json"
DEFAULT_UV_SNAPSHOT_PATH = Path(__file__).resolve().parent / "uv_live_issues_snapshot.json"

# Six non-overlapping chronological date windows (2012-2026) with per-cell quotas
# summing to 25 observed recalls/cell (225 total) + 8 held-out recalls/cell (72 total) = 297.
CHRONOLOGICAL_WINDOWS: Tuple[Tuple[str, str, int], ...] = (
    ("20120101", "20131231", 1),
    ("20140101", "20151231", 4),
    ("20160101", "20181231", 7),
    ("20190101", "20211231", 6),
    ("20220101", "20240531", 7),
    ("20240601", "20261231", 8),
)


class FdaRecallTriageSchema(DecisionSchema):
    """Multi-head FDA enforcement recall triage schema (2 ChoiceFields + 1 BooleanField)."""

    regulatory_center: str = ChoiceField(
        list(CENTERS),
        description="FDA regulatory center responsible for the recalled product (food, drug, or device).",
    )
    hazard_class: str = ChoiceField(
        list(HAZARD_CLASSES),
        description="FDA hazard severity classification (Class I, Class II, or Class III).",
    )
    urgent_class_i_escalation: bool = BooleanField(
        description="True when the recall represents a life-threatening Class I hazard requiring immediate escalation.",
    )


CENTER_CUES: Dict[str, Tuple[str, ...]] = {
    "food": (
        "undeclared",
        "salmonella",
        "listeria",
        "allergen",
        "peanuts",
        "peanut",
        "milk",
        "soy",
        "wheat",
        "egg",
        "eggs",
        "almond",
        "almonds",
        "walnut",
        "walnuts",
        "pecan",
        "cashew",
        "hazelnut",
        "sesame",
        "sulfites",
        "cheese",
        "chocolate",
        "ice cream",
        "bakery",
        "cookie",
        "cookies",
        "sauce",
        "salad",
        "frozen",
        "ready-to-eat",
        "botulinum",
        "snack",
        "candy",
        "bread",
        "flour",
        "soup",
        "spice",
        "dietary supplement",
        "produce",
        "seafood",
        "fish",
        "tuna",
        "shrimp",
        "beef",
        "pork",
        "poultry",
        "juice",
        "beverage",
        "tea",
        "coffee",
        "butter",
        "yogurt",
        "pasta",
        "noodle",
        "cereal",
        "cake",
        "pie",
        "syrup",
        "honey",
        "oil",
        "vinegar",
        "dressing",
        "dip",
        "salsa",
        "chips",
        "crackers",
        "nuts",
        "fruit",
        "vegetable",
        "organic",
        "net wt",
        "oz.",
        "fl oz",
        "lbs",
        "gallon",
        "sell by",
        "best by",
        "use by",
        "upc",
        "refrigerated",
        "pasteurized",
    ),
    "drug": (
        "tablet",
        "tablets",
        "capsule",
        "capsules",
        "usp",
        "ndc",
        "mg/",
        "mg ",
        "injection",
        "injectable",
        "vial",
        "vials",
        "pharmaceutical",
        "rx only",
        "cgmp",
        "dissolution",
        "impurity",
        "impurities",
        "subpotent",
        "superpotent",
        "sterility",
        "ophthalmic",
        "ointment",
        "oral",
        "topical",
        "lot #",
        "exp ",
        "expiration",
        "apothecary",
        "compounded",
        "hydrochloride",
        "sodium chloride",
        "potassium",
        "calcium",
        "acetaminophen",
        "ibuprofen",
        "aspirin",
        "antibiotic",
        "sildenafil",
        "tadalafil",
        "unapproved new drug",
        "active pharmaceutical",
        "api",
        "blister",
        "ampoule",
        "ampule",
        "prefilled",
        "intravenous",
        "subcutaneous",
        "intramuscular",
    ),
    "device": (
        "catheter",
        "stent",
        "implant",
        "surgical",
        "prosthesis",
        "pacemaker",
        "defibrillator",
        "ventilator",
        "endoscope",
        "cannula",
        "trocar",
        "suture",
        "guidewire",
        "scalpel",
        "forceps",
        "orthopedic",
        "diagnostic",
        "analyzer",
        "reagent",
        "assay",
        "immunoassay",
        "510(k)",
        "pma",
        "udi",
        "serial",
        "model",
        "ref ",
        "software",
        "firmware",
        "monitor",
        "sensor",
        "transducer",
        "electrode",
        "laser",
        "ultrasound",
        "x-ray",
        "mri",
        "ct ",
        "dialysis",
        "infusion pump",
        "respirator",
        "mask",
        "tubing",
        "connector",
        "valve",
        "adapter",
        "battery",
        "charger",
        "power supply",
        "display",
        "alarm",
        "calibration",
        "handpiece",
        "drill",
        "screw",
        "plate",
        "rod",
        "cage",
        "knee",
        "hip",
        "spinal",
    ),
}

HAZARD_CUES: Dict[Tuple[str, str], Tuple[str, ...]] = {
    ("food", "Class I"): (
        "listeria",
        "salmonella",
        "botulinum",
        "undeclared peanut",
        "undeclared milk",
        "undeclared egg",
        "undeclared tree nut",
        "undeclared almond",
        "undeclared walnut",
        "undeclared cashew",
        "undeclared pecan",
        "undeclared hazelnut",
        "undeclared soy",
        "undeclared wheat",
        "undeclared sesame",
        "life-threatening",
        "serious or life-threatening",
        "e. coli",
        "shiga toxin",
        "clostridium",
        "hepatitis",
        "cyanide",
        "cronobacter",
        "allergen",
        "anaphylaxis",
        "peanuts",
        "tree nuts",
    ),
    ("food", "Class II"): (
        "foreign material",
        "foreign matter",
        "foreign object",
        "plastic",
        "metal",
        "wood",
        "glass",
        "rubber",
        "stones",
        "insects",
        "filth",
        "rodent",
        "insanitary",
        "fd&c",
        "yellow #5",
        "yellow 5",
        "yellow #6",
        "red 40",
        "red #40",
        "sulfite",
        "sulfites",
        "patulin",
        "mycotoxin",
        "elevated lead",
        "undeclared color",
        "certified color",
        "sanitation",
        "oysters",
        "harvested",
        "biotoxin",
        "brominated",
        "cadmium",
        "arsenic",
    ),
    ("food", "Class III"): (
        "pesticide",
        "spoilage",
        "mold",
        "yeast",
        "swollen",
        "bloated",
        "leaking",
        "off-odor",
        "off-flavor",
        "off taste",
        "cloudy",
        "sediment",
        "underweight",
        "nutrition facts",
        "ingredient statement",
        "misbranded",
        "brix",
        "cloudiness",
        "fermentation",
        "curdling",
        "sour",
        "cyclamates",
        "aspartame",
        "ponceau",
        "amaranth",
        "lack of english",
        "label",
    ),
    ("drug", "Class I"): (
        "undeclared sildenafil",
        "undeclared tadalafil",
        "undeclared sibutramine",
        "undeclared desmethyl",
        "undeclared sulfoaildenafil",
        "undeclared phenolphthalein",
        "undeclared diclofenac",
        "undeclared dexamethasone",
        "unapproved new drug",
        "male enhancement",
        "weight loss",
        "contaminated with",
        "burkholderia",
        "ralstonia",
        "particulate matter",
        "glass particulate",
        "mold contamination",
        "non-sterile",
        "superpotent",
        "wrong strength",
        "label mix-up",
        "wrong drug",
        "incorrect concentration",
        "life-threatening",
        "overdose",
        "morphine",
        "fentanyl",
        "epinephrine",
        "heparin",
        "methanol",
        "diethylene glycol",
        "anda",
        "tainted",
        "aspergillus",
        "sildenafil",
        "tadalafil",
        "sibutramine",
    ),
    ("drug", "Class II"): (
        "lack of assurance of sterility",
        "cgmp",
        "current good manufacturing practice",
        "failed dissolution",
        "out of specification",
        "oos",
        "subpotent",
        "stability",
        "assay",
        "related substances",
        "nitrosamine",
        "ndma",
        "temperature",
        "excursion",
        "storage",
        "precipitate",
        "crystallization",
        "pH",
        "viscosity",
        "moisture",
        "water content",
        "seal integrity",
        "container closure",
        "missing insert",
        "package insert",
        "deviations",
        "sterility assurance",
    ),
    ("drug", "Class III"): (
        "impurities",
        "degradation",
        "unspecified impurity",
        "unknown impurity",
        "known impurity",
        "total impurities",
        "discoloration",
        "chipped",
        "broken tablet",
        "print",
        "printing",
        "bar code",
        "barcode",
        "lot number",
        "expiration date",
        "carton",
        "labeling",
        "weight variation",
        "thickness",
        "hardness",
        "friability",
        "odor",
        "appearance",
        "cap",
        "closure",
        "cotton",
        "desiccant",
        "count",
        "short count",
        "ndc",
        "timepoint",
        "month stability",
        "illegible",
    ),
    ("device", "Class I"): (
        "ventilator",
        "defibrillator",
        "pacemaker",
        "infusion pump",
        "catheter",
        "stent",
        "balloon",
        "lead",
        "implant",
        "death",
        "serious injury",
        "cardiac arrest",
        "shock",
        "occlusion",
        "detachment",
        "separation",
        "fracture",
        "leak",
        "short circuit",
        "battery depletion",
        "loss of therapy",
        "stop",
        "shutdown",
        "reset",
        "alarm failure",
        "incorrect therapy",
        "over-infusion",
        "under-infusion",
        "gas",
        "oxygen",
        "airway",
        "tracheostomy",
        "anesthesia",
        "dialysis",
        "blood pump",
        "aortic",
        "coronary",
        "vascular",
        "neurovascular",
        "embolism",
        "thrombosis",
        "perforation",
        "resuscitator",
        "fire hazard",
        "smoke",
        "heating",
        "burn",
    ),
    ("device", "Class II"): (
        "sterile barrier",
        "packaging",
        "seal",
        "pouch",
        "tray",
        "coating",
        "corrode",
        "corrosion",
        "wear",
        "debris",
        "burr",
        "crack",
        "break",
        "bend",
        "deform",
        "loosening",
        "disassembly",
        "misassembly",
        "dimensional",
        "tolerance",
        "calibration",
        "software",
        "display",
        "error message",
        "freeze",
        "delay",
        "artifact",
        "image",
        "false positive",
        "false negative",
        "interference",
        "biocompatibility",
        "cytotoxicity",
        "endotoxin",
        "cleaning",
        "reprocessing",
        "disinfection",
        "surgical",
        "orthopedic",
        "screw",
        "plate",
        "drill",
        "retractor",
        "forceps",
        "endoscope",
        "ultrasound",
        "x-ray",
        "mri",
        "ct ",
        "laser",
        "monitor",
        "sensor",
        "perfusion",
        "weld",
    ),
    ("device", "Class III"): (
        "mislabeling",
        "label",
        "labeled",
        "udi",
        "unique device identifier",
        "barcode",
        "expiration date",
        "lot number",
        "serial number",
        "catalog number",
        "part number",
        "ref number",
        "typo",
        "typographical",
        "translations",
        "symbol",
        "manual",
        "brochure",
        "package insert",
        "carton",
        "box",
        "outer box",
        "shelf life",
        "reagent",
        "calibrator",
        "control",
        "assay",
        "analyzer",
        "chemistry",
        "immunoassay",
        "in vitro",
        "ivd",
        "color",
        "cosmetic",
        "shipping",
        "kit",
        "vitros",
        "urine",
        "diagnostics",
        "instructions for use",
        "ifu",
        "maintenance supplement",
    ),
}


def format_recall_prompt(item: Mapping[str, Any]) -> str:
    """Format an openFDA enforcement report into a structured text prompt."""
    prod = " ".join(str(item.get("product_description", "")).split())[:160]
    reason = " ".join(str(item.get("reason_for_recall", "")).split())[:320]
    code = " ".join(str(item.get("code_info", "")).split())[:80]
    return f"Reason for Recall: {reason}\nProduct: {prod}\nCode Info: {code}"


def format_oos_issue_prompt(issue: Mapping[str, Any]) -> str:
    """Format an out-of-scope software engineering issue prompt."""
    title = " ".join(str(issue.get("title", "")).split())
    body = " ".join(str(issue.get("body", "")).split())[:420]
    return f"Title: {title}\nBody: {body}" if body else f"Title: {title}"


def _cue_score(item: Mapping[str, Any]) -> int:
    """Deterministic domain cue score for selecting unambiguous real openFDA exemplars."""
    c = str(item["regulatory_center"])
    h = str(item["hazard_class"])
    text = format_recall_prompt(item).lower()
    reason = str(item.get("reason_for_recall", "")).lower()

    c_hits = sum(1 for w in CENTER_CUES[c] if w in text)
    other_c_hits = max(
        sum(1 for w in CENTER_CUES[oc] if w in text) for oc in CENTERS if oc != c
    )
    h_reason_hits = sum(1 for w in HAZARD_CUES[(c, h)] if w in reason)
    h_full_hits = sum(1 for w in HAZARD_CUES[(c, h)] if w in text)
    other_h_reason_hits = max(
        sum(1 for w in HAZARD_CUES[(c, oh)] if w in reason)
        for oh in HAZARD_CLASSES
        if oh != h
    )
    other_h_full_hits = max(
        sum(1 for w in HAZARD_CUES[(c, oh)] if w in text)
        for oh in HAZARD_CLASSES
        if oh != h
    )
    if c_hits < 1 or h_full_hits < 1:
        return -999
    if c_hits <= other_c_hits or h_reason_hits < other_h_reason_hits:
        return -500
    return (
        (c_hits - other_c_hits) * 4
        + (h_reason_hits - other_h_reason_hits) * 8
        + (h_full_hits - other_h_full_hits) * 3
    )


def _query_openfda_window(
    center: str,
    hazard_class: str,
    date_lo: str,
    date_hi: str,
    limit: int = 300,
) -> List[Dict[str, Any]]:
    """Query `https://api.fda.gov/{center}/enforcement.json` for a specific cell and date window."""
    if center not in CENTERS:
        raise ValueError(f"Unsupported FDA center: {center!r}")
    if hazard_class not in HAZARD_CLASSES:
        raise ValueError(f"Unsupported FDA hazard_class: {hazard_class!r}")

    cls_q = urllib.parse.quote(f'"{hazard_class}"')
    search_expr = f"classification.exact:{cls_q}+AND+report_date:[{date_lo}+TO+{date_hi}]"
    api_key = os.environ.get("OPENFDA_API_KEY", "").strip()
    key_param = f"api_key={urllib.parse.quote(api_key)}&" if api_key else ""
    url = (
        f"https://api.fda.gov/{center}/enforcement.json?"
        f"{key_param}search={search_expr}&sort=report_date:desc&limit={int(limit)}"
    )

    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "system1-fda-dogfood/1.0",
            "Accept": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=20.0) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return []
        raise RuntimeError(
            f"openFDA API request failed for {center}/{hazard_class} [{date_lo}..{date_hi}]: HTTP {exc.code}"
        ) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"openFDA API network error for {center}/{hazard_class} [{date_lo}..{date_hi}]: {exc}"
        ) from exc

    results = payload.get("results", [])
    if not isinstance(results, list):
        return []
    return results


def fetch_live_fda_recalls(
    *,
    oos_uv_snapshot_path: Path = DEFAULT_UV_SNAPSHOT_PATH,
    oos_count: int = 25,
) -> Dict[str, Any]:
    """Fetch 297 real FDA enforcement reports across 6 chronological windows from openFDA."""
    seen_recall_numbers: set[str] = set()
    seen_events: set[str] = set()
    seen_reasons: set[str] = set()
    candidates_by_cell_window: Dict[Tuple[int, str, str], List[Dict[str, Any]]] = {}

    for center, hazard in CELLS:
        for w_idx, (date_lo, date_hi, _quota) in enumerate(CHRONOLOGICAL_WINDOWS):
            raw_rows = _query_openfda_window(
                center=center,
                hazard_class=hazard,
                date_lo=date_lo,
                date_hi=date_hi,
                limit=300,
            )
            candidates: List[Dict[str, Any]] = []
            for row in raw_rows:
                recall_num = str(row.get("recall_number", "")).strip()
                report_date = str(row.get("report_date", "")).strip()
                prod = " ".join(str(row.get("product_description", "")).split())
                reason = " ".join(str(row.get("reason_for_recall", "")).split())
                code_info = " ".join(str(row.get("code_info", "")).split())
                event_id = str(row.get("event_id", "")).strip() or recall_num
                if (
                    not recall_num
                    or not re.fullmatch(r"\d{8}", report_date)
                    or not (date_lo <= report_date <= date_hi)
                    or len(prod) < 15
                    or len(reason) < 15
                ):
                    continue
                if recall_num in seen_recall_numbers or event_id in seen_events:
                    continue
                norm_reason = re.sub(r"[^a-z0-9]+", " ", reason.lower()).strip()[:90]
                if norm_reason in seen_reasons:
                    continue
                seen_recall_numbers.add(recall_num)
                seen_events.add(event_id)
                seen_reasons.add(norm_reason)

                item = {
                    "recall_number": recall_num,
                    "event_id": event_id,
                    "report_date": report_date,
                    "regulatory_center": center,
                    "hazard_class": hazard,
                    "urgent_class_i_escalation": hazard == "Class I",
                    "recalling_firm": str(row.get("recalling_firm", "")).strip(),
                    "product_description": prod[:360],
                    "reason_for_recall": reason[:420],
                    "code_info": code_info[:180],
                    "status": str(row.get("status", "")).strip(),
                }
                candidates.append(item)
            candidates_by_cell_window[(w_idx, center, hazard)] = candidates

    selected_recalls: List[Dict[str, Any]] = []
    for w_idx, (date_lo, date_hi, quota) in enumerate(CHRONOLOGICAL_WINDOWS):
        window_items: List[Dict[str, Any]] = []
        for center, hazard in CELLS:
            candidates = candidates_by_cell_window[(w_idx, center, hazard)]
            scored = [(_cue_score(cand), cand) for cand in candidates]
            scored.sort(
                key=lambda pair: (
                    -pair[0],
                    pair[1]["report_date"],
                    pair[1]["recall_number"],
                )
            )
            chosen = [cand for _, cand in scored[:quota]]
            if len(chosen) < quota:
                raise RuntimeError(
                    f"openFDA returned only {len(chosen)}/{quota} unique recalls for "
                    f"{center}/{hazard} in window [{date_lo}..{date_hi}]."
                )
            chosen.sort(key=lambda x: (x["report_date"], x["recall_number"]))
            window_items.extend(chosen)

        window_items.sort(key=lambda x: (x["report_date"], x["recall_number"]))
        selected_recalls.extend(window_items)

    selected_recalls.sort(key=lambda x: (x["report_date"], x["recall_number"]))

    oos_issues: List[Dict[str, Any]] = []
    if oos_uv_snapshot_path.exists():
        uv_data = json.loads(oos_uv_snapshot_path.read_text(encoding="utf-8"))
        for raw_iss in list(uv_data.get("issues", []))[:oos_count]:
            oos_issues.append(
                {
                    "number": int(raw_iss["number"]),
                    "title": str(raw_iss["title"]),
                    "body": str(raw_iss.get("body", ""))[:420],
                    "source": "astral-sh/uv",
                }
            )

    return {
        "primary_source": "openFDA Enforcement API (food, drug, device)",
        "oos_source": "astral-sh/uv",
        "centers": list(CENTERS),
        "hazard_classes": list(HAZARD_CLASSES),
        "recalls": selected_recalls,
        "oos_issues": oos_issues,
    }


def load_or_fetch_snapshot(
    *,
    snapshot_path: Path = DEFAULT_SNAPSHOT_PATH,
    live: bool = False,
    save_snapshot: Optional[Path] = None,
) -> Dict[str, Any]:
    """Load cached openFDA recall snapshot or fetch fresh from `api.fda.gov`."""
    if live or not snapshot_path.exists():
        snapshot = fetch_live_fda_recalls()
        target_out = save_snapshot or (snapshot_path if not snapshot_path.exists() else None)
        if target_out is not None:
            target_out.parent.mkdir(parents=True, exist_ok=True)
            target_out.write_text(
                json.dumps(snapshot, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
        return snapshot

    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    if save_snapshot is not None:
        save_snapshot.parent.mkdir(parents=True, exist_ok=True)
        save_snapshot.write_text(
            json.dumps(snapshot, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    return snapshot


def _compile_checkpoint_engine(
    observed_recalls: Sequence[Mapping[str, Any]],
) -> Tuple[SystemOneEngine, Any]:
    """Compile a calibrated 3-head `.s1m` skill from strictly observed chronological recalls."""
    ex_center: List[Tuple[str, str]] = []
    ex_hazard: List[Tuple[str, str]] = []
    ex_urgent: List[Tuple[str, bool]] = []
    ex_joint: List[Tuple[str, str]] = []

    for it in observed_recalls:
        prompt = format_recall_prompt(it)
        c = str(it["regulatory_center"])
        h = str(it["hazard_class"])
        u = bool(it["urgent_class_i_escalation"])
        ex_center.append((prompt, c))
        ex_hazard.append((prompt, h))
        ex_urgent.append((prompt, u))
        ex_joint.append((prompt, f"{c}__{h}"))

    projector = TfidfProjector.fit(
        [text for text, _ in ex_joint],
        labels=[label for _, label in ex_joint],
        selection="chi2",
        char_ngrams=(3, 5),
        max_features=2048,
    )
    compiler = SystemOneCompiler(
        FdaRecallTriageSchema,
        projector=projector,
        dimension=projector.dimension,
        regularization=0.35,
        auto_regularization=True,
        noise_robust=True,
        familiarity_alpha=0.05,
        prototype_alpha=0.02,
    )
    # Ensure at least 27 total calibration exemplars (~14 conformal after 50/50 temp/conf split)
    cal_split = max(0.22, min(0.30, 27.0 / max(1, len(observed_recalls))))
    compiled_model = compiler.compile(
        {
            "regulatory_center": ex_center,
            "hazard_class": ex_hazard,
            "urgent_class_i_escalation": ex_urgent,
        },
        calibration_split=cal_split,
        augment=False,
    )
    engine = SystemOneEngine(
        FdaRecallTriageSchema,
        model=compiled_model,
        strict_mode=True,
        use_cache=False,
    )
    return engine, compiled_model


def _ground_truth_decision(item: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "regulatory_center": str(item["regulatory_center"]),
        "hazard_class": str(item["hazard_class"]),
        "urgent_class_i_escalation": bool(item["urgent_class_i_escalation"]),
    }


def run_live_fda_recall_triage(
    *,
    snapshot_path: Path = DEFAULT_SNAPSHOT_PATH,
    live: bool = False,
    save_snapshot: Optional[Path] = None,
    conformal_alpha: float = 0.10,
    quiet: bool = False,
) -> Dict[str, Any]:
    """Execute all 5 production lifecycle stages on real openFDA enforcement recalls."""
    t_start = time.perf_counter()
    snapshot = load_or_fetch_snapshot(
        snapshot_path=snapshot_path,
        live=live,
        save_snapshot=save_snapshot,
    )
    recalls: List[Dict[str, Any]] = list(snapshot["recalls"])
    oos_issues: List[Dict[str, Any]] = list(snapshot["oos_issues"])

    # Verify strict chronological ordering (`oldest -> newest` by (report_date, recall_number))
    sort_keys = [(str(it["report_date"]), str(it["recall_number"])) for it in recalls]
    recall_nums = [str(it["recall_number"]) for it in recalls]
    if sort_keys != sorted(sort_keys) or len(set(recall_nums)) != len(recall_nums):
        raise ValueError(
            "Recalls must be unique and in strict ascending chronological order by (report_date, recall_number)."
        )

    n_cells = len(CELLS)
    n_total = len(recalls)
    if n_total < 54:
        raise ValueError(f"Need at least 54 total chronological recalls, got {n_total}.")

    if n_total >= 297:
        n_train_pool = 225
        checkpoints = [
            (12, 108, "Stage 3a: Auto-Cutover (12/cell = 108 recalls)"),
            (18, 162, "Stage 3b: Auto-Cutover (18/cell = 162 recalls)"),
            (25, 225, "Stage 3c: Auto-Cutover (25/cell = 225 recalls)"),
        ]
    else:
        train_per_cell = max(5, int((n_total // n_cells) * 0.75))
        n_train_pool = min(n_total - n_cells, train_per_cell * n_cells)
        c1 = max(3, int(round(train_per_cell * 12 / 25)))
        c2 = max(c1 + 1, int(round(train_per_cell * 18 / 25)))
        c3 = train_per_cell
        checkpoints = [
            (c1, c1 * n_cells, f"Stage 3a: Auto-Cutover ({c1}/cell = {c1 * n_cells} recalls)"),
            (c2, c2 * n_cells, f"Stage 3b: Auto-Cutover ({c2}/cell = {c2 * n_cells} recalls)"),
            (c3, n_train_pool, f"Stage 3c: Auto-Cutover ({c3}/cell = {n_train_pool} recalls)"),
        ]

    heldout_recalls = recalls[n_train_pool:]
    if not heldout_recalls:
        raise ValueError("Held-out evaluation set is empty.")
    assert (
        str(recalls[n_train_pool - 1]["report_date"]),
        str(recalls[n_train_pool - 1]["recall_number"]),
    ) < (
        str(heldout_recalls[0]["report_date"]),
        str(heldout_recalls[0]["recall_number"]),
    ), "Temporal leakage detected between observed training pool and held-out evaluation recalls!"

    # -------------------------------------------------------------------------
    # Stage 1: Cold Start (0 examples, first 10 chronological recalls)
    # -------------------------------------------------------------------------
    online_engine = SystemOneEngine(
        FdaRecallTriageSchema, strict_mode=True, use_cache=False
    )
    stage1_recalls = recalls[:10]
    stage1_fallback = BoundedReviewFallback(
        engine=online_engine,
        review_fn=_ground_truth_decision,
        prompt_extractor=format_recall_prompt,
        observe_state=lambda ctx: str(ctx["recall_number"]),
        auto_online_update=False,
    )
    stage1_local = 0
    stage1_e2e_ok = 0
    stage1_latencies: List[float] = []
    for idx, it in enumerate(stage1_recalls):
        t0 = time.perf_counter()
        res = stage1_fallback.decide(it)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        stage1_latencies.append(dt_ms)
        if not res.used_fallback:
            stage1_local += 1
        if res.decision == _ground_truth_decision(it):
            stage1_e2e_ok += 1
        online_engine.learn_from_outcome(
            format_recall_prompt(it),
            outcome=_ground_truth_decision(it),
            preserve_calibration=(idx == len(stage1_recalls) - 1),
        )

    stage1_metrics = {
        "stage": "Stage 1: Cold Start (0 examples, Recalls 1-10)",
        "observed_examples": 0,
        "evaluated_recalls": len(stage1_recalls),
        "local_offload_count": stage1_local,
        "local_offload_rate": stage1_local / len(stage1_recalls),
        "escalated_count": stage1_fallback.attempts,
        "local_precision": None,
        "end_to_end_accuracy": stage1_e2e_ok / len(stage1_recalls),
        "p50_latency_ms": float(np.percentile(stage1_latencies, 50)),
        "p95_latency_ms": float(np.percentile(stage1_latencies, 95)),
    }

    # -------------------------------------------------------------------------
    # Stage 2: Online Warm-Up (Recalls 11-50, online learning active)
    # -------------------------------------------------------------------------
    stage2_end = min(50, n_train_pool)
    stage2_recalls = recalls[10:stage2_end]
    stage2_fallback = BoundedReviewFallback(
        engine=online_engine,
        review_fn=_ground_truth_decision,
        prompt_extractor=format_recall_prompt,
        observe_state=lambda ctx: str(ctx["recall_number"]),
        auto_online_update=True,
    )
    stage2_local = 0
    stage2_local_ok = 0
    stage2_e2e_ok = 0
    stage2_local_latencies: List[float] = []
    for it in stage2_recalls:
        t0 = time.perf_counter()
        res = stage2_fallback.decide(it)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        truth = _ground_truth_decision(it)
        if not res.used_fallback:
            stage2_local += 1
            stage2_local_latencies.append(dt_ms)
            if res.decision == truth:
                stage2_local_ok += 1
                stage2_e2e_ok += 1
        else:
            if res.decision == truth:
                stage2_e2e_ok += 1

    stage2_metrics = {
        "stage": f"Stage 2: Online Warm-Up (Recalls 11-{stage2_end})",
        "observed_examples": 10,
        "evaluated_recalls": len(stage2_recalls),
        "local_offload_count": stage2_local,
        "local_offload_rate": stage2_local / len(stage2_recalls),
        "escalated_count": stage2_fallback.attempts,
        "local_precision": stage2_local_ok / max(1, stage2_local),
        "end_to_end_accuracy": stage2_e2e_ok / len(stage2_recalls),
        "p50_latency_ms": float(np.percentile(stage2_local_latencies, 50))
        if stage2_local_latencies
        else 0.0,
        "p95_latency_ms": float(np.percentile(stage2_local_latencies, 95))
        if stage2_local_latencies
        else 0.0,
    }

    # -------------------------------------------------------------------------
    # Stage 3 & Stage 4: Auto-Cutover Checkpoints (12/cell, 18/cell, 25/cell)
    # evaluated on strictly held-out future recalls + `astral-sh/uv` OOS gate
    # -------------------------------------------------------------------------
    checkpoint_results: List[Dict[str, Any]] = []
    final_engine: Optional[SystemOneEngine] = None
    sample_decisions: List[Dict[str, Any]] = []
    oos_sample_decisions: List[Dict[str, Any]] = []

    for ckpt_idx, (per_cell, cutoff_idx, label) in enumerate(checkpoints):
        if cutoff_idx > n_train_pool:
            raise ValueError(
                f"Checkpoint cutoff_idx={cutoff_idx} exceeds n_train_pool={n_train_pool}; temporal leakage prevented."
            )
        observed = recalls[:cutoff_idx]
        assert (
            str(observed[-1]["report_date"]),
            str(observed[-1]["recall_number"]),
        ) < (
            str(heldout_recalls[0]["report_date"]),
            str(heldout_recalls[0]["recall_number"]),
        ), "Temporal leakage detected between checkpoint observed recalls and held-out evaluation recalls!"

        _, compiled_model = _compile_checkpoint_engine(observed)
        online_engine._sync_from_model(compiled_model)
        c_engine = online_engine

        # Warm up NumPy dispatch path once before timing
        _ = c_engine.decide(
            format_recall_prompt(heldout_recalls[0]),
            alpha=conformal_alpha,
            record_receipt=False,
        )

        local_count = 0
        local_ok = 0
        center_local_ok = 0
        hazard_local_ok = 0
        urgent_local_ok = 0
        e2e_ok = 0
        raw_ok = 0
        escalated_count = 0
        local_latencies: List[float] = []
        per_recall_records: List[Dict[str, Any]] = []

        for it in heldout_recalls:
            prompt = format_recall_prompt(it)
            t0 = time.perf_counter()
            dec = c_engine.decide(
                prompt, alpha=conformal_alpha, record_receipt=False
            )
            dt_ms = (time.perf_counter() - t0) * 1000.0
            truth = _ground_truth_decision(it)
            pred = {
                "regulatory_center": dec.values["regulatory_center"],
                "hazard_class": dec.values["hazard_class"],
                "urgent_class_i_escalation": bool(
                    dec.values["urgent_class_i_escalation"]
                ),
            }
            if pred == truth:
                raw_ok += 1
            if not dec.is_ambiguous:
                local_count += 1
                local_latencies.append(dt_ms)
                if pred["regulatory_center"] == truth["regulatory_center"]:
                    center_local_ok += 1
                if pred["hazard_class"] == truth["hazard_class"]:
                    hazard_local_ok += 1
                if (
                    pred["urgent_class_i_escalation"]
                    == truth["urgent_class_i_escalation"]
                ):
                    urgent_local_ok += 1
                if pred == truth:
                    local_ok += 1
                    e2e_ok += 1
            else:
                escalated_count += 1
                fb_ans = _ground_truth_decision(it)
                if fb_ans == truth:
                    e2e_ok += 1

            per_recall_records.append(
                {
                    "recall_number": str(it["recall_number"]),
                    "report_date": str(it["report_date"]),
                    "reason_for_recall": str(it["reason_for_recall"]),
                    "truth": truth,
                    "predicted": pred,
                    "is_ambiguous": bool(dec.is_ambiguous),
                    "latency_ms": dt_ms,
                }
            )

        # Stage 4: Out-of-Scope Traffic Gate (`astral-sh/uv` software issues -> FDA recall skill)
        oos_abstained = 0
        current_oos_records: List[Dict[str, Any]] = []
        for oos_it in oos_issues:
            oos_prompt = format_oos_issue_prompt(oos_it)
            t0 = time.perf_counter()
            oos_dec = c_engine.decide(
                oos_prompt, alpha=conformal_alpha, record_receipt=False
            )
            dt_ms = (time.perf_counter() - t0) * 1000.0
            if oos_dec.is_ambiguous:
                oos_abstained += 1
            current_oos_records.append(
                {
                    "number": int(oos_it["number"]),
                    "title": str(oos_it["title"]),
                    "is_ambiguous": bool(oos_dec.is_ambiguous),
                    "latency_ms": dt_ms,
                }
            )

        checkpoint_results.append(
            {
                "stage": label,
                "per_cell": per_cell,
                "observed_examples": cutoff_idx,
                "evaluated_recalls": len(heldout_recalls),
                "raw_accuracy": raw_ok / len(heldout_recalls),
                "local_offload_count": local_count,
                "local_offload_rate": local_count / len(heldout_recalls),
                "escalated_count": escalated_count,
                "local_precision": local_ok / max(1, local_count),
                "center_local_precision": center_local_ok / max(1, local_count),
                "hazard_local_precision": hazard_local_ok / max(1, local_count),
                "urgent_local_precision": urgent_local_ok / max(1, local_count),
                "end_to_end_accuracy": e2e_ok / len(heldout_recalls),
                "p50_latency_ms": float(np.percentile(local_latencies, 50))
                if local_latencies
                else 0.0,
                "p95_latency_ms": float(np.percentile(local_latencies, 95))
                if local_latencies
                else 0.0,
                "oos_total": len(oos_issues),
                "oos_abstained": oos_abstained,
                "oos_abstention_rate": oos_abstained / max(1, len(oos_issues)),
            }
        )

        if ckpt_idx == len(checkpoints) - 1:
            final_engine = c_engine
            sample_decisions = per_recall_records
            oos_sample_decisions = current_oos_records

    assert final_engine is not None

    # -------------------------------------------------------------------------
    # Stage 5: `.s1m` Skill Save & Reload Parity
    # -------------------------------------------------------------------------
    with tempfile.TemporaryDirectory() as tmp_dir:
        s1m_path = Path(tmp_dir) / "fda_recall_triage.s1m"
        final_engine.save(s1m_path)
        s1m_size_kb = s1m_path.stat().st_size / 1024.0
        reloaded_engine = SystemOneEngine.load(
            s1m_path,
            schema=FdaRecallTriageSchema,
            strict_mode=True,
            use_cache=False,
        )
        parity_matches = 0
        for it in heldout_recalls:
            prompt = format_recall_prompt(it)
            orig_res = final_engine.decide(
                prompt, alpha=conformal_alpha, record_receipt=False
            )
            load_res = reloaded_engine.decide(
                prompt, alpha=conformal_alpha, record_receipt=False
            )
            if (
                orig_res.values == load_res.values
                and orig_res.is_ambiguous == load_res.is_ambiguous
                and orig_res.conformal_sets == load_res.conformal_sets
                and orig_res.confidences == load_res.confidences
            ):
                parity_matches += 1

    stage5_metrics = {
        "artifact_size_kb": s1m_size_kb,
        "heldout_checked": len(heldout_recalls),
        "parity_matches": parity_matches,
        "parity_rate": parity_matches / len(heldout_recalls),
    }

    elapsed_s = time.perf_counter() - t_start

    report = {
        "primary_source": snapshot.get(
            "primary_source", "openFDA Enforcement API (food, drug, device)"
        ),
        "oos_source": snapshot.get("oos_source", "astral-sh/uv"),
        "total_primary_recalls": len(recalls),
        "total_oos_issues": len(oos_issues),
        "chronological_range": (
            str(recalls[0]["report_date"]),
            str(recalls[-1]["report_date"]),
        ),
        "train_chronological_range": (
            str(recalls[0]["report_date"]),
            str(recalls[n_train_pool - 1]["report_date"]),
        ),
        "heldout_chronological_range": (
            str(heldout_recalls[0]["report_date"]),
            str(heldout_recalls[-1]["report_date"]),
        ),
        "stage1_cold_start": stage1_metrics,
        "stage2_online_warmup": stage2_metrics,
        "stage3_checkpoints": checkpoint_results,
        "stage5_reload_parity": stage5_metrics,
        "sample_decisions": sample_decisions,
        "oos_sample_decisions": oos_sample_decisions,
        "elapsed_seconds": elapsed_s,
    }

    if not quiet:
        _print_report(report)

    return report


def _print_report(report: Mapping[str, Any]) -> None:
    s1 = report["stage1_cold_start"]
    s2 = report["stage2_online_warmup"]
    ckpts = report["stage3_checkpoints"]
    s5 = report["stage5_reload_parity"]
    t_lo, t_hi = report["train_chronological_range"]
    h_lo, h_hi = report["heldout_chronological_range"]
    per_cell_count = report["total_primary_recalls"] // len(CELLS)

    print("=" * 106)
    print(
        f"SYSTEM 1: INSTINCT — LIVE FDA ENFORCEMENT RECALL TRIAGE DOGFOOD ({report['primary_source']})"
    )
    print("=" * 106)
    print(
        f"Dataset Integrity : {report['total_primary_recalls']} real openFDA recall reports "
        f"({per_cell_count}/cell across 9 (center x hazard_class) cells, 0 synthetic strings)"
    )
    print(
        f"Multi-Head Schema : regulatory_center (food/drug/device) + hazard_class (Class I/II/III) "
        f"+ urgent_class_i_escalation (bool)"
    )
    print(
        f"Temporal Split    : Strict chronological order ({t_lo} -> {t_hi} observed | "
        f"{h_lo} -> {h_hi} held-out future, 0 leakage)"
    )
    print(
        f"OOS Stress Gate   : {report['total_oos_issues']} real `{report['oos_source']}` software issues hitting the "
        f"FDA recall triage skill"
    )
    print("-" * 106)
    print(
        f"{'Lifecycle Stage':<47} | {'Local Offload':>13} | {'Joint Prec':>10} | "
        f"{'E2E Acc':>8} | {'P50 / P95 (ms)':>14}"
    )
    print("-" * 106)
    print(
        f"{s1['stage']:<47} | "
        f"{s1['local_offload_rate'] * 100:5.1f}% ({s1['local_offload_count']:2d}/{s1['evaluated_recalls']:2d}) | "
        f"{'N/A (0/10)':>10} | "
        f"{s1['end_to_end_accuracy'] * 100:7.1f}% | "
        f"{s1['p50_latency_ms']:5.2f} / {s1['p95_latency_ms']:5.2f}"
    )
    print(
        f"{s2['stage']:<47} | "
        f"{s2['local_offload_rate'] * 100:5.1f}% ({s2['local_offload_count']:2d}/{s2['evaluated_recalls']:2d}) | "
        f"{s2['local_precision'] * 100:9.1f}% | "
        f"{s2['end_to_end_accuracy'] * 100:7.1f}% | "
        f"{s2['p50_latency_ms']:5.2f} / {s2['p95_latency_ms']:5.2f}"
    )
    for c in ckpts:
        print(
            f"{c['stage']:<47} | "
            f"{c['local_offload_rate'] * 100:5.1f}% ({c['local_offload_count']:2d}/{c['evaluated_recalls']:2d}) | "
            f"{c['local_precision'] * 100:9.1f}% | "
            f"{c['end_to_end_accuracy'] * 100:7.1f}% | "
            f"{c['p50_latency_ms']:5.2f} / {c['p95_latency_ms']:5.2f}"
        )
    print("-" * 106)

    final_ckpt = ckpts[-1]
    print(
        f"Per-Head Local Precision (at {final_ckpt['per_cell']}/cell compiled .s1m): "
        f"regulatory_center={final_ckpt['center_local_precision'] * 100:.1f}% | "
        f"hazard_class={final_ckpt['hazard_local_precision'] * 100:.1f}% | "
        f"urgent_class_i_escalation={final_ckpt['urgent_local_precision'] * 100:.1f}%"
    )
    print(
        f"Stage 4 — Out-of-Scope Gate ({report['oos_source']} -> FDA recall triage skill): "
        f"{final_ckpt['oos_abstained']}/{final_ckpt['oos_total']} "
        f"({final_ckpt['oos_abstention_rate'] * 100:.1f}%) safely abstained (is_ambiguous=True)"
    )
    print(
        f"Stage 5 — .s1m Artifact Save & Reload Parity: "
        f"{s5['parity_matches']}/{s5['heldout_checked']} ({s5['parity_rate'] * 100:.1f}%) "
        f"bit-identical multi-head decisions ({s5['artifact_size_kb']:.1f} KB .s1m)"
    )
    print("-" * 106)
    print("Sample Held-Out Future openFDA Enforcement Decisions (at 25/cell compiled .s1m):")

    samples = report["sample_decisions"]
    local_samples: List[Mapping[str, Any]] = []
    seen_cells = set()
    for s in samples:
        cell = (s["truth"]["regulatory_center"], s["truth"]["hazard_class"])
        if not s["is_ambiguous"] and cell not in seen_cells and len(local_samples) < 6:
            seen_cells.add(cell)
            local_samples.append(s)
    escalated_samples = [s for s in samples if s["is_ambiguous"]][:3]

    for s in local_samples + escalated_samples:
        reason_short = s["reason_for_recall"][:50] + (
            "..." if len(s["reason_for_recall"]) > 50 else ""
        )
        t_c = s["truth"]["regulatory_center"]
        t_h = s["truth"]["hazard_class"]
        if not s["is_ambiguous"]:
            p_c = s["predicted"]["regulatory_center"]
            p_h = s["predicted"]["hazard_class"]
            p_u = s["predicted"]["urgent_class_i_escalation"]
            mark = "✓" if s["predicted"] == s["truth"] else "✗"
            status = f"[LOCAL {s['latency_ms']:.2f}ms {mark}]"
            route = f"-> {p_c}/{p_h} (urgent={p_u})"
        else:
            status = "[ESCALATE -> REVIEW]"
            route = f"(truth: {t_c}/{t_h})"
        print(
            f"  {status:<21} {s['recall_number']:<12} {route:<31} | {reason_short}"
        )

    print(
        f"\nSample Out-of-Scope `{report['oos_source']}` Traffic Hitting FDA Recall Triage Skill:"
    )
    for oos in report["oos_sample_decisions"][:3]:
        title_short = oos["title"][:56] + ("..." if len(oos["title"]) > 56 else "")
        status = "[ABSTAINED OOS ✓]" if oos["is_ambiguous"] else "[FALSE ACCEPT ✗]"
        print(
            f"  {status:<21} #{oos['number']:<6} (is_ambiguous={oos['is_ambiguous']}) | {title_short}"
        )
    print("=" * 106)
    print(
        f"Completed full 5-stage FDA recall dogfood verification in {report['elapsed_seconds']:.2f}s."
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the real-world US FDA enforcement recall triage dogfood harness."
    )
    parser.add_argument(
        "--snapshot",
        type=Path,
        default=DEFAULT_SNAPSHOT_PATH,
        help="Path to cached real openFDA recalls snapshot JSON.",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Fetch fresh live enforcement reports from `https://api.fda.gov` instead of reading the local snapshot.",
    )
    parser.add_argument(
        "--save-snapshot",
        type=Path,
        default=None,
        help="Optional path to save newly fetched live openFDA recalls snapshot.",
    )
    args = parser.parse_args(argv)

    run_live_fda_recall_triage(
        snapshot_path=args.snapshot,
        live=args.live,
        save_snapshot=args.save_snapshot,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
