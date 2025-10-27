from inspect_ai import Epochs, Task, task
from inspect_ai.dataset import Sample, csv_dataset, FieldSpec
from inspect_ai.model import GenerateConfig
from inspect_ai.solver import prompt_template, solver, generate
from inspect_ai.agent import react
from inspect_ai.scorer import match, mean_score, pass_at
from inspect_ai.tool import bash, python

from typing import Iterable, List
import math
from latex_scorer import latex_exact

MATH_OLYMPIAD_ENGLISH_PATH = "data/T_human_English_Olympiad_Mathematics.csv"

MATH_PROMPT_TEMPLATE = """
Solve the following math problem step by step. The last line of your
response should be of the form "ANSWER: $ANSWER" (without quotes)
where $ANSWER is the answer to the problem.

{prompt}
""".strip()


def strip_list_wrapper(s: str) -> str:
    s = s.strip()
    if (s.startswith("['") and s.endswith("']")) or (
        s.startswith('["') and s.endswith('"]')
    ):
        return s[2:-2]
    return s


def record_to_sample(record):
    return Sample(
        input=record["question"],
        target=strip_list_wrapper(record["final_answer"]),
        id=record["id"],
        metadata={
            "subfield": record["subfield"],
            "solution": record["solution"],
            "is_multiple_answer": record["is_multiple_answer"],
            "unit": record["unit"],
            "answer_type": record["answer_type"],
            "error": record["error"],
            "label": record["label"],
            "source": record["source"],
            "support_auto_scoring": record["support_auto_scoring"],
            "classification": record["classification"],
            "T_human": record["T_human"],
            "exam": record["exam"],
        },
    )


def load_math_olympiad_english() -> list[Sample]:
    dataset = csv_dataset(MATH_OLYMPIAD_ENGLISH_PATH, record_to_sample)
    return dataset


def select_by_T_human_bins(
    samples: Iterable["Sample"],
    n_bins: int = 8,
    per_bin: int = 1,
    log_space: bool = True,
) -> List["Sample"]:
    """
    Deterministically select `per_bin` samples from each T_human bin.
    Bins are equal-width in log10(T_human) by default.

    Returns:
        List[Sample]: concatenated selections in ascending bin order.
    """
    # --- parse and filter ---
    items = []
    for s in samples:
        md = getattr(s, "metadata", {}) or {}
        t = md.get("T_human", None)
        try:
            t = float(t)
        except (TypeError, ValueError):
            continue
        if not (t and t > 0):
            continue
        x = math.log10(t) if log_space else t
        items.append((s, t, x))

    if not items:
        return []

    # stable id getter for tie-breaking
    def _sid(sample: "Sample"):
        sid = getattr(sample, "id", None)
        if sid is None:
            md = getattr(sample, "metadata", {}) or {}
            sid = md.get("id", "")
        return str(sid)

    # sort by transformed time then id (makes bin edges + picks deterministic)
    items.sort(key=lambda it: (it[2], _sid(it[0])))

    # --- build bins (equal-width) ---
    xs = [x for _, _, x in items]
    xmin, xmax = min(xs), max(xs)
    if xmin == xmax:
        n_eff = 1
        edges = [xmin, xmax]
    else:
        n_eff = n_bins
        step = (xmax - xmin) / n_eff
        edges = [xmin + i * step for i in range(n_eff)] + [xmax]

    bins: list[list[tuple["Sample", float, float]]] = [[] for _ in range(n_eff)]
    span = (xmax - xmin) or 1.0
    for s, t, x in items:
        idx = int((x - xmin) / span * n_eff)
        if idx >= n_eff:  # rightmost edge case
            idx = n_eff - 1
        bins[idx].append((s, t, x))

    # --- deterministic pick: earliest by T_human, then id ---
    selected: List["Sample"] = []
    for bucket in bins:
        if not bucket:
            continue
        bucket.sort(key=lambda it: (it[1], _sid(it[0])))
        selected.extend([it[0] for it in bucket[:per_bin]])

    return selected


@solver
def agent(attempts: int = 1):
    return react(
        tools=[bash(timeout=180), python(timeout=180)],
        attempts=attempts,
    )


@task
def math_olympiad_english():
    dataset = load_math_olympiad_english()
    subset = select_by_T_human_bins(dataset, n_bins=8, per_bin=1, log_space=True)
    return Task(
        dataset=subset,
        # solver=[agent(), prompt_template(MATH_PROMPT_TEMPLATE)],
        solver=[prompt_template(MATH_PROMPT_TEMPLATE), generate()],
        scorer=latex_exact(),
        epochs=Epochs(8, [mean_score(), pass_at(8)]),
        # config=GenerateConfig(temperature=0.7),
    )
