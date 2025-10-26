from inspect_ai import Task, task
from inspect_ai.dataset import Sample, csv_dataset, FieldSpec
from inspect_ai.solver import prompt_template, solver
from inspect_ai.agent import react
from inspect_ai.scorer import match
from inspect_ai.tool import bash, python

MATH_OLYMPIAD_ENGLISH_PATH = "data/T_human_English_Olympiad_Mathematics.csv"

MATH_PROMPT_TEMPLATE = """
Solve the following math problem step by step. The last line of your
response should be of the form "ANSWER: $ANSWER" (without quotes)
where $ANSWER is the answer to the problem.

{prompt}
""".strip()

def load_math_olympiad_english() -> list[Sample]:
    dataset = csv_dataset(
        MATH_OLYMPIAD_ENGLISH_PATH,
        FieldSpec(
            input="question",
            target="final_answer",
            id="id",
            metadata=[
                "subfield","solution","is_multiple_answer","unit","answer_type",
                "error","label","source","support_auto_scoring","classification",
                "T_human","exam",
            ],
        ),
    )
    return dataset

@solver
def agent(attempts: int = 3):
    return react(
        tools=[bash(timeout=180), python(timeout=180)],
        attempts=attempts,
    )

@task
def math_olympiad_english():
    all_samples = load_math_olympiad_english()
    short_samples = [s for s in all_samples if float(s.metadata["T_human"]) < 3]
    return Task(
        dataset=short_samples,
        solver=[agent(), prompt_template(MATH_PROMPT_TEMPLATE)],
        scorer=match(numeric=True),
    )
