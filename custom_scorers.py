import re
from inspect_ai.scorer import scorer, Score, CORRECT, INCORRECT, accuracy, stderr
from auto_scoring_judge import AutoScoringJudge

ANSWER_RE = re.compile(r'(?mi)^[ \t]*\**\s*answer\s*:\s*(.*?)\s*\**[ \t]*$')

_LATEX_SUBS = [
    # Basic normalization
    (r"\\{2}", r"\\"),                        # collapse double backslashes (\\frac -> \frac)
    (r"[\u200b\u200c\u200d\u2060\uFEFF]", ""), # remove zero-width / BOM characters

    # Strip math-mode delimiters
    (r"\$+", ""),                            # remove $...$
    (r"\\\(|\\\)|\\\[|\\\]", ""),            # remove \( \) \[ \]

    # Remove sizing and spacing
    (r"\\left|\\right", ""),                 # sizing
    (r"\\big|\\Big|\\bigg|\\Bigg", ""),      # more sizing variants
    (r"\\,|\\!|\\;|\\:|\\ ", ""),            # thin/micro spaces
    (r"~", ""),                              # non-breaking space

    # Common fraction & style normalization
    (r"\\tfrac", r"\\frac"),                 # \tfrac -> \frac
    (r"\\dfrac", r"\\frac"),                 # \dfrac -> \frac
    (r"\\cfrac", r"\\frac"),                 # \cfrac -> \frac
    (r"\\displaystyle", ""),                 # remove displaystyle

    # Remove decorations
    (r"\\boxed\{([^{}]*)\}", r"\1"),         # \boxed{...} -> ...
    (r"\\textstyle", ""),                    # remove textstyle
    (r"\\scriptstyle", ""),                  # remove scriptstyle

    # Common LaTeX wrappers and font commands
    (r"\\mathrm\{([^{}]*)\}", r"\1"),        # \mathrm{x} -> x
    (r"\\mathbf\{([^{}]*)\}", r"\1"),        # \mathbf{x} -> x
    (r"\\mathit\{([^{}]*)\}", r"\1"),        # \mathit{x} -> x
    (r"\\text\{([^{}]*)\}", r"\1"),          # \text{x} -> x
    (r"\\operatorname\{([^{}]*)\}", r"\1"),  # \operatorname{sin} -> sin

    # Standardize known symbols
    (r"\\approx|\\simeq|\\sim", "="),        # treat approximate as equal
    (r"∶", ":"),                             # normalize ratio colon
    (r"，", ","),                             # Chinese comma
    (r"；", ";"),                             # Chinese semicolon
    (r"−", "-"),                             # minus sign variant
    (r"×", r"\\times"),                      # × -> \times
    (r"÷", r"\\div"),                        # ÷ -> \div
    (r"∞", r"\\infty"),                      # ∞ -> \infty
    (r"\\emptyset", r"\\varnothing"),        # unify empty set
    (r"\\phi", r"\\varphi"),                 # unify phi variants
    (r"\\varepsilon", r"\\epsilon"),         # unify epsilon
    (r"\\Re", r"\\operatorname{Re}"),        # ensure consistent real/imag
    (r"\\Im", r"\\operatorname{Im}"),        # same for imaginary

    # Normalize interval unions
    (r"U", r"\\cup"),                        # plain U -> \cup
    (r"∪", r"\\cup"),                        # Unicode union

    # Fix bracket spacing and redundancy
    (r"\s+", ""),                            # collapse whitespace
    (r"\(\s*", "("),                         # trim spaces inside brackets
    (r"\s*\)", ")"),
    (r"\[\s*", "["),
    (r"\s*\]", "]"),
]

def canonical_latex(s: str) -> str:
    s = (s or "").strip()
    for pat, rep in _LATEX_SUBS:
        s = re.sub(pat, rep, s)
    m = re.fullmatch(r'\(?\s*(-?)(\d+)\s*/\s*(\d+)\s*\)?', s)
    if m:
        sign, num, den = m.groups()
        s = ( "-" if sign else "" ) + rf"\frac{{{num}}}{{{den}}}"

    return s

def extract_answer_line(text: str) -> str:
    matches = ANSWER_RE.findall(text or "")
    if matches:
        return matches[-1]
    lines = [ln for ln in (text or "").splitlines() if ln.strip()]
    return lines[-1].strip() if lines else ""

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
        model_output_raw = extract_answer_line(completion)
        ground_truth_raw = getattr(target, "text", "") or ""

        # Canonicalize both sides first

        model_output = canonical_latex(model_output_raw)
        ground_truth = canonical_latex(ground_truth_raw)

        if model_output == ground_truth:
            return Score(value=CORRECT, answer=model_output)

        result = scorer.judge(ground_truth, model_output, precision)
        return Score(
            value=CORRECT if result else INCORRECT,
            answer=model_output
        )
    
    return score
