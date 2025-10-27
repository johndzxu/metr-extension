import re
from inspect_ai.scorer import scorer, Score, CORRECT, INCORRECT, accuracy, stderr
from auto_scoring_judge import AutoScoringJudge

ANSWER_RE = re.compile(r'(?mi)^[ \t]*\**\s*answer\s*:\s*(.*?)\s*\**[ \t]*$')

_LATEX_SUBS = [
    (r"\$+", ""),                         # strip $...$
    (r"\\\(|\\\)|\\\[|\\\]", ""),        # strip \( \) \[ \]
    (r"\\left|\\right", ""),             # sizing
    (r"\\,|\\!|\\;|\\:", ""),            # thin spaces
    (r"\\tfrac", r"\\frac"),             # \tfrac -> \frac
    (r"\\dfrac", r"\\frac"),             # \dfrac -> \frac
    (r"\\boxed\{([^{}]*)\}", r"\1"),     # \boxed{...} -> ...
    (r"\s+", ""),                        # collapse whitespace
]

def canonical_latex(s: str) -> str:
    s = (s or "").strip()
    for pat, rep in _LATEX_SUBS:
        s = re.sub(pat, rep, s)
    return s

def extract_answer_line(text: str) -> str:
    matches = ANSWER_RE.findall(text or "")
    if matches:
        return matches[-1]
    # fallback: last non-empty line
    lines = [ln for ln in (text or "").splitlines() if ln.strip()]
    return lines[-1] if lines else ""

@scorer(metrics=[accuracy(), stderr()])
def latex_exact():
    async def score(state, target):
        completion = getattr(getattr(state, "output", None), "completion", "") or ""
        pred_raw = extract_answer_line(completion)
        gold_raw = getattr(target, "text", "") or ""

        pred = canonical_latex(pred_raw)
        gold = canonical_latex(gold_raw)

        return Score(
            value=CORRECT if pred == gold else INCORRECT,
            answer=pred_raw  # optional: shows what was extracted
        )
    return score

@scorer(metrics=[accuracy(), stderr()])
def olympiadbench_scorer(precision: float = 1e-8):
    scorer = AutoScoringJudge()
    async def score(state, target):
        completion = getattr(getattr(state, "output", None), "completion", "") or ""
        model_output = extract_answer_line(completion)
        ground_truth = getattr(target, "text", "") or ""

        result = scorer.judge(ground_truth, model_output, precision)
        return Score(
            value=CORRECT if result else INCORRECT,
            answer=model_output
        )
    
    return score
