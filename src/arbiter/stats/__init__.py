"""Statistics for deciding whether a noisy eval suite regressed.

Nothing in this package imports anything outside the standard library. That is
deliberate: the error guarantees are the product, so the code that provides them
should be readable in one sitting and testable without a numerical stack.
"""

from .bayes import BetaPosterior, prob_worse, prob_worse_by
from .confseq import ConfidenceSequence, ConfSeqSpec, radius
from .multiple import FdrResult, benjamini_hochberg, e_bh, holm, p_to_e
from .paired import PairedSpec, PairedSprt, PairOutcome
from .power import PairedPlan, SuitePlan, TaskPlan, fixed_sample_size, paired_plan, plan_suite
from .special import beta_ppf, betainc, norm_cdf, norm_ppf
from .sprt import Sprt, SprtSpec, Verdict

__all__ = [
    "BetaPosterior",
    "ConfSeqSpec",
    "ConfidenceSequence",
    "FdrResult",
    "PairOutcome",
    "PairedPlan",
    "PairedSpec",
    "PairedSprt",
    "Sprt",
    "SprtSpec",
    "SuitePlan",
    "TaskPlan",
    "Verdict",
    "benjamini_hochberg",
    "beta_ppf",
    "betainc",
    "e_bh",
    "fixed_sample_size",
    "holm",
    "norm_cdf",
    "norm_ppf",
    "p_to_e",
    "paired_plan",
    "plan_suite",
    "prob_worse",
    "prob_worse_by",
    "radius",
]
