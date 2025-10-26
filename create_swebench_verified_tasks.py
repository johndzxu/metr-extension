import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from datasets import load_dataset


def load_metr_data(jsonl_path: Path) -> dict[str, dict[str, Any]]:
    """Load metadata from swe_bench_runs.jsonl."""
    metadata_map: dict[str, dict[str, Any]] = {}
    
    with open(jsonl_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            task_id = data.get('task_id', '')
            metadata_map[task_id] = {
                'human_minutes': data.get('human_minutes'),
                'score_cont': data.get('score_cont'),
                'score_binarized': data.get('score_binarized'),
                'task_family': data.get('task_family'),
                'equal_task_weight': data.get('equal_task_weight'),
                'invsqrt_task_weight': data.get('invsqrt_task_weight')
            }
    
    return metadata_map


def extract_repo_category(repo: str) -> str:
    """Extract category from repo name """

    return repo.split('/')[0]


def format_problem_statement(row: dict[str, Any]) -> str:
    
    problem_statement = row.get('problem_statement')
    hints = row.get('hints_text')
    
    hints = ""
    if hints and hints.strip():
        hints = f"\n\n**Hints:**\n{hints}\n"
    
    fail_to_pass = row.get('FAIL_TO_PASS')
    pass_to_pass = row.get('PASS_TO_PASS')
    
    tests = "\n\n**Test Information:**\n"
    if fail_to_pass:
        tests += f"Tests that should PASS after your fix ({len(fail_to_pass)} tests):\n"
        for test in fail_to_pass[:5]: 
            tests += f"  - {test}\n"
        if len(fail_to_pass) > 5:
            tests += f"  ... and {len(fail_to_pass) - 5} more\n"
    
    if pass_to_pass:
        tests += f"\nTests that must continue to PASS ({len(pass_to_pass)} tests):\n"
        for test in pass_to_pass[:3]:  
            tests += f"  - {test}\n"
        if len(pass_to_pass) > 3:
            tests += f"  ... and {len(pass_to_pass) - 3} more\n"
    
    return f"""Repository: {row['repo']}
Instance ID: {row['instance_id']}
Base Commit: {row['base_commit']}
Environment Setup Commit: {row.get('environment_setup_commit')}

**Problem Statement:**
{problem_statement}{hints}{tests}

**Your Task:**
Fix this issue by modifying the codebase. You have access to:
- Code execution (Python, Bash)
- File system operations to read and modify files
- Web search for documentation
- Git operations to examine the repository

**Instructions:**
1. Clone the repository and checkout the base commit
2. Set up the development environment
3. Analyze the issue thoroughly
4. Locate the relevant code
5. Implement a fix
6. Run the failing tests to verify they now pass
7. Run the existing tests to ensure no regressions
8. Explain your changes

**Success Criteria:**
- All FAIL_TO_PASS tests must pass
- All PASS_TO_PASS tests must continue to pass
- No new test failures introduced
"""


def format_target(row: dict[str, Any]) -> str:
    
    fail_to_pass = row.get('FAIL_TO_PASS')
    pass_to_pass = row.get('PASS_TO_PASS')

    return f"""**Success Criteria:**

1. **Issue Resolution**: The reported issue must be fixed
   - The bug/problem described in the issue is resolved
   - The fix addresses the root cause, not just symptoms

2. **Test Validation**: All specified tests must pass
   - FAIL_TO_PASS tests ({len(fail_to_pass)}): Currently failing tests must pass after fix
   - PASS_TO_PASS tests ({len(pass_to_pass)}): Existing passing tests must continue to pass
   - No regression: No new test failures introduced

"""


def create_inspect_task(row: dict[str, Any], metadata: dict[str, Any]) -> dict[str, Any]:
    """Create an inspect task from a SWE-bench row and metadata."""
    input_text = format_problem_statement(row)
    target_text = format_target(row)
    
    fail_to_pass = row.get('FAIL_TO_PASS')
    pass_to_pass = row.get('PASS_TO_PASS')

    task_metadata = {
        "instance_id": row['instance_id'],
        "repo": row['repo'],
        "base_commit": row['base_commit'],
        "environment_setup_commit": row.get('environment_setup_commit', ''),
        "task_source": "SWE-Bench-Verified",
        "created_at": row.get('created_at', ''),
        "version": row.get('version', ''),
        "has_hints": bool(row.get('hints_text', '').strip()),
        "human_minutes": metadata.get('human_minutes'),
        "baseline_score_cont": metadata.get('score_cont'),
        "baseline_score_binarized": metadata.get('score_binarized'),
        # Test information
        "fail_to_pass_tests": fail_to_pass,
        "pass_to_pass_tests": pass_to_pass,
        "num_fail_to_pass": len(fail_to_pass),
        "num_pass_to_pass": len(pass_to_pass),
    }

    return {
        "input": input_text,
        "target": target_text,
        "metadata": task_metadata
    }


def main() -> None:

    dataset = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")
    
    # Load metadata from swe_bench_runs.jsonl
    jsonl_path = Path(__file__).parent / "metr_data" / "external" / "swe_bench_runs.jsonl"
    metadata_map = load_metr_data(jsonl_path)

    
    # Group by repository category
    categories= defaultdict(list)
    
    for row in dataset:
        repo = row['repo']
        category = extract_repo_category(repo)
        categories[category].append(row)
    

    output_dir = Path(__file__).parent / "inspect_datasets" / "swebench_verified"

    
    all_tasks = []
    
    for category, samples in sorted(categories.items()):
        selected_samples = samples[:5]
        
        # Create inspect tasks for this category

        for sample in selected_samples:
            instance_id = sample['instance_id']

            metadata = metadata_map.get(instance_id)
            
            task = create_inspect_task(sample, metadata)

            task['metadata']['category'] = category

            all_tasks.append(task)
        

    output_file = output_dir / "combined_samples.json"
    with open(output_file, 'w') as f:
        json.dump(all_tasks, f, indent=2)
    


if __name__ == "__main__":
    main()

