# -*- coding: utf-8 -*-
"""Prices and the partial budget.

The economics follow the DIFM / OFPE convention: net return = crop price x yield - input price x rate, per acre, with
every cost that does not change with the rate left out (they cancel between VR and flat). A variable-rate
implementation cost (prescription, imagery, controller, agronomist time) can be entered per acre and is subtracted from
the VR gain only.

Rates and prices must be in the same basis:
    * seed       rate lb/ac, price $/lb
    * urea       rate lb product/ac, price $/lb product              (basis = "product")
    * nitrogen   rate lb N/ac, price $/lb N = $/lb urea / 0.46        (basis = "nutrient")
Grain prices are entered per bushel; $/tonne is converted with the crop's bushel weight.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict

LB_PER_TONNE = 2204.62262
LB_PER_KG = 2.20462262
AC_PER_HA = 2.47105381
KG_HA_TO_LB_AC = LB_PER_KG / AC_PER_HA             # 0.892179: lb/ac per kg/ha
RATE_UNITS = ("lb/ac", "kg/ha")

# An input's name becomes a column suffix (rate_<name>, target_<name>, design_<name>) and a formula term, so it must
# be an identifier, and it must not collide with the columns the engine makes itself. The app checks the names as
# they are typed; Prices.from_dict checks them again for a project.json or a config given to engine.run by hand.
INPUT_NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")
RESERVED_INPUT_NAMES = ("rate", "design_rate", "target", "yield", "zone", "ndvi", "plot_id", "treatment", "block")


def check_input_names(names):
    """Raise ValueError for a duplicate (case-blind), a non-identifier or a reserved input name."""
    seen = set()
    for n in names:
        s = str(n or "")
        if not INPUT_NAME_RE.match(s):
            raise ValueError(f"Input name {s!r} is not valid: letters, digits and _ only, starting with a letter")
        if s.lower() in RESERVED_INPUT_NAMES:
            raise ValueError(f"Input name {s!r} is reserved ({', '.join(RESERVED_INPUT_NAMES)}, in any case): name "
                             "the input by what was tested (seed, urea, ...)")
        if s.lower() in seen:
            raise ValueError(f"Two inputs are named {s!r}: every input needs a name of its own")
        seen.add(s.lower())


def rate_unit_factor(unit):
    """Multiplier that takes a rate in `unit` to lb/ac, the unit of every rate inside the engine."""
    u = (unit or "lb/ac").strip().lower().replace(" ", "")
    if u in ("lb/ac", "lb/acre", "lbs/ac", "lbs/acre", ""):
        return 1.0
    if u in ("kg/ha", "kgs/ha"):
        return KG_HA_TO_LB_AC
    raise ValueError(f"Unknown rate unit {unit!r}: use one of {', '.join(RATE_UNITS)}")

# bushel weights used to convert $/t <-> $/bu (Canadian Grain Commission standard bushel weights)
BUSHEL_LB = {"wheat": 60.0, "durum": 60.0, "barley": 48.0, "oats": 34.0, "canola": 50.0, "peas": 60.0,
             "lentils": 60.0, "flax": 56.0, "corn": 56.0, "soybeans": 60.0, "rye": 56.0}

# nutrient fraction of common products (lb nutrient per lb product)
PRODUCTS = {
    "seed": {"label": "Seed", "fraction": 1.0, "unit": "lb/ac"},
    "urea 46-0-0": {"label": "Urea 46-0-0", "fraction": 0.46, "unit": "lb product/ac"},
    "anhydrous 82-0-0": {"label": "Anhydrous ammonia 82-0-0", "fraction": 0.82, "unit": "lb product/ac"},
    "UAN 28-0-0": {"label": "UAN 28-0-0", "fraction": 0.28, "unit": "lb product/ac"},
    "ESN 44-0-0": {"label": "ESN 44-0-0", "fraction": 0.44, "unit": "lb product/ac"},
    "MAP 11-52-0": {"label": "MAP 11-52-0 (P2O5)", "fraction": 0.52, "unit": "lb product/ac"},
    "other": {"label": "Other", "fraction": 1.0, "unit": "unit/ac"},
}


def bushel_lb(crop: str) -> float:
    c = (crop or "").lower()
    for k, v in BUSHEL_LB.items():
        if k in c:
            return v
    return 60.0


def per_tonne_to_per_bu(price_t: float, crop: str) -> float:
    return float(price_t) * bushel_lb(crop) / LB_PER_TONNE


def per_bu_to_per_tonne(price_bu: float, crop: str) -> float:
    return float(price_bu) * LB_PER_TONNE / bushel_lb(crop)


@dataclass
class InputPrice:
    """One tested input. `price_per_lb_product` is what the grower pays; `rate_basis` says whether the trial rates are
    lb of product (seed, urea as applied) or lb of nutrient (N)."""
    name: str = "seed"
    product: str = "seed"
    price_per_lb_product: float = 0.22
    rate_basis: str = "product"          # "product" | "nutrient"
    rate_unit: str = "lb/ac"
    status_quo: float | None = None      # the grower's flat rate (ask, never infer)

    @property
    def fraction(self) -> float:
        return float(PRODUCTS.get(self.product, PRODUCTS["other"])["fraction"])

    @property
    def price_per_rate_unit(self) -> float:
        """$ per one unit of the trial's rate column."""
        if self.rate_basis == "nutrient" and self.fraction > 0:
            return float(self.price_per_lb_product) / self.fraction
        return float(self.price_per_lb_product)


@dataclass
class Prices:
    crop: str = "wheat"
    crop_price_bu: float = 9.25
    currency: str = "CAD"
    vr_cost_ac: float = 0.0              # extra cost of running VR, $/ac (subtracted from VR gains only)
    inputs: list = field(default_factory=lambda: [InputPrice()])
    crop_price_range: tuple = (8.5, 10.0)
    input_price_range_pct: tuple = (-30.0, 30.0)       # the grid spans crop AND input price by this much
    source: str = ""
    as_of: str = ""

    def input(self, name):
        for i in self.inputs:
            if i.name == name:
                return i
        raise KeyError(name)

    def to_dict(self):
        d = asdict(self)
        d["inputs"] = [asdict(i) for i in self.inputs]
        return d

    @classmethod
    def from_dict(cls, d):
        d = dict(d or {})
        ins = [InputPrice(**i) for i in d.pop("inputs", [])] or [InputPrice()]
        check_input_names([i.name for i in ins])
        for i in ins:
            rate_unit_factor(i.rate_unit)       # an unknown unit is refused here, before any rate is read
        for k in ("crop_price_range", "input_price_range_pct"):
            if k in d and d[k] is not None:
                d[k] = tuple(d[k])
        return cls(inputs=ins, **{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def net_return(yield_bu_ac, rate, crop_price_bu, input_price_per_unit):
    """Partial net return, $/ac."""
    return crop_price_bu * yield_bu_ac - input_price_per_unit * rate
