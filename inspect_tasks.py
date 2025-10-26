"""
This file contains Inspect AI task definitions 
"""
import json
import os
from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.solver import generate
from inspect_ai.agent import react
from inspect_ai.scorer import model_graded_fact
from inspect_ai.tool import bash, python, web_search
from pathlib import Path


os.environ["HF_TOKEN"] = "your_token_here" 

# Configuration: Choose solver based on model capabilities
USE_FUNCTION_CALLING = False  # Set to True for OpenAI/Anthropic models, False for HuggingFace

def get_solver():
    """Return appropriate solver based on configuration."""
    if USE_FUNCTION_CALLING:
        return react(tools=[bash(), python(), web_search()])
    else:
        return generate()

def get_scorer_template():
    """Return appropriate scorer template based on solver."""
    if USE_FUNCTION_CALLING:
        return """
        Evaluate the code implementation based on:
        1. Issue Resolution: Is the reported bug/issue completely fixed?
        2. Test Results: Do the FAIL_TO_PASS tests now pass?
        3. No Regressions: Do PASS_TO_PASS tests still pass?
        4. Code Quality: Are changes minimal, clean, and follow project conventions?
        5. Correctness: Are edge cases handled properly?
        
        Provide a score from 0.0 (issue not resolved) to 1.0 (perfect fix).
        Format: ANSWER: [score] followed by explanation.
        """
    else:
        return """
        Evaluate the proposed solution based on:
        1. Problem Understanding: Does the solution address the root cause?
        2. Code Quality: Are the proposed changes clean and minimal?
        3. Test Consideration: Does the solution consider the test requirements?
        4. Framework Patterns: Does it follow project conventions?
        5. Completeness: Is the solution comprehensive?
        
        Score 0.0-1.0. Format: ANSWER: [score] followed by explanation.
        """


def load_all_swebench_verified() -> list[Sample]:
    """Load all SWE-Bench Verified samples"""
    combined_file = Path("inspect_datasets/swebench_verified/combined_samples.json")
    
    with open(combined_file, 'r') as f:
        data = json.load(f)
    
    return [Sample(**sample) for sample in data]

@task
def swebench_verified():
    """SWE-Bench Verified - All 53 verified tasks from 12 categories.
    
    This dataset combines problem descriptions from HuggingFace's SWE-bench Verified
    with human baseline completion times from METR's evaluation data.
    """
    return Task(
        dataset=load_all_swebench_verified(),
        solver=get_solver(),
        scorer=model_graded_fact(template=get_scorer_template())
    )

@task
def swebench_verified_django():
    """SWE-Bench Verified - Django framework tasks (5 tasks)."""
    all_samples = load_all_swebench_verified()
    django_samples = [sample for sample in all_samples if sample.metadata.get('category') == 'django']
    
    return Task(
        dataset=django_samples,
        solver=get_solver(),
        scorer=model_graded_fact(template=get_scorer_template())
    )

@task
def swebench_verified_matplotlib():
    """SWE-Bench Verified - Matplotlib visualization library tasks (5 tasks)."""
    all_samples = load_all_swebench_verified()
    matplotlib_samples = [sample for sample in all_samples if sample.metadata.get('category') == 'matplotlib']
    
    return Task(
        dataset=matplotlib_samples,
        solver=get_solver(),
        scorer=model_graded_fact(template=get_scorer_template())
    )

@task
def swebench_verified_scikit_learn():
    """SWE-Bench Verified - Scikit-learn machine learning library tasks (5 tasks)."""
    all_samples = load_all_swebench_verified()
    scikit_learn_samples = [sample for sample in all_samples if sample.metadata.get('category') == 'scikit-learn']
    
    return Task(
        dataset=scikit_learn_samples,
        solver=get_solver(),
        scorer=model_graded_fact(template=get_scorer_template())
    )

@task
def swebench_verified_sympy():
    """SWE-Bench Verified - SymPy symbolic mathematics tasks (5 tasks)."""
    all_samples = load_all_swebench_verified()
    sympy_samples = [sample for sample in all_samples if sample.metadata.get('category') == 'sympy']
    
    return Task(
        dataset=sympy_samples,
        solver=get_solver(),
        scorer=model_graded_fact(template=get_scorer_template())
    )

@task
def swebench_verified_pytest():
    """SWE-Bench Verified - Pytest testing framework tasks (5 tasks)."""
    all_samples = load_all_swebench_verified()
    pytest_samples = [sample for sample in all_samples if sample.metadata.get('category') == 'pytest-dev']
    
    return Task(
        dataset=pytest_samples,
        solver=get_solver(),
        scorer=model_graded_fact(template=get_scorer_template())
    )

@task
def swebench_verified_sphinx():
    """SWE-Bench Verified - Sphinx documentation generator tasks (5 tasks)."""
    all_samples = load_all_swebench_verified()
    sphinx_samples = [sample for sample in all_samples if sample.metadata.get('category') == 'sphinx-doc']
    
    return Task(
        dataset=sphinx_samples,
        solver=get_solver(),
        scorer=model_graded_fact(template=get_scorer_template())
    )
