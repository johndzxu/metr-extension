import argparse
import json
import logging
import pathlib
from collections import defaultdict
from typing import Any


def load_all_samples(input_path: pathlib.Path) -> list[dict[str, Any]]:

    if input_path.is_file():
        # Load from summaries.json
        with open(input_path) as f:
            return json.load(f)
    elif input_path.is_dir():
        # Load from individual sample files
        samples = []
        for sample_file in sorted(input_path.glob("*_epoch_*.json")):
            with open(sample_file) as f:
                samples.append(json.load(f))
        return samples



def calculate_task_weights(
    samples: list[dict[str, Any]], 
    weight_type: str = "subfield"
) -> dict[str, dict[str, float]]:
    """Calculate task weights based on diversity.

    """
    # Count tasks per category
    category_counts: dict[str, int] = defaultdict(int)
    task_categories: dict[str, str] = {}
    
    for sample in samples:
        task_id = sample["id"]
        metadata = sample.get("metadata", {})
        
        if weight_type == "subfield":
            category = metadata.get("subfield", "Unknown")
        elif weight_type == "exam":
            category = metadata.get("exam", "Unknown")
        else:
            raise ValueError(f"Unknown weight_type: {weight_type}")
        
        task_categories[task_id] = category
        category_counts[category] += 1
    
    # Calculate weights
    total_tasks = len(samples)
    weights = {}
    
    for task_id, category in task_categories.items():
        equal_weight = 1.0 / total_tasks
        # Inverse square root weighting for diversity
        invsqrt_weight = 1.0 / (category_counts[category] ** 0.5)
        
        weights[task_id] = {
            "equal_task_weight": equal_weight,
            f"invsqrt_task_weight_{weight_type}": invsqrt_weight,
            "category": category,
            "category_count": category_counts[category]
        }
    
    # Normalize 
    total_invsqrt = sum(w[f"invsqrt_task_weight_{weight_type}"] for w in weights.values())
    for w in weights.values():
        w[f"invsqrt_task_weight_{weight_type}"] /= total_invsqrt
    
    return weights


def convert_score(score_value: str) -> tuple[float, int]:

    if score_value == "C":
        return 1.0, 1
    elif score_value == "I":
        return 0.0, 0


def extract_model_name(model_usage: dict[str, Any]) -> str:

    if not model_usage:
        return "Failed"  
    
    full_name = list(model_usage.keys())[0]
    
    if full_name.startswith("hf-inference-providers/"):
        full_name = full_name.replace("hf-inference-providers/", "")
    elif full_name.startswith("openrouter/"):
        full_name = full_name.replace("openrouter/", "")
    
    if "/" in full_name:
        parts = full_name.split("/")
        return parts[-1]  # e.g., "DeepSeek-R1"
    
    return full_name


def convert_sample_to_runs(
    sample: dict[str, Any],
    weights_subfield: dict[str, dict[str, float]],
    weights_exam: dict[str, dict[str, float]]
) -> list[dict[str, Any]]:
    """Convert a single Inspect sample to one or more METR run entries.
    """

    task_id = sample["id"]
    epoch = sample.get("epoch", 1)
    metadata = sample.get("metadata", {})
    

    scores = sample.get("scores", {})
    score_dict = scores.get("match", {}) or scores.get("latex_exact", {})
    score_value = score_dict.get("value", "I")
    score_cont, score_binarized = convert_score(score_value)
    

    model_usage = sample.get("model_usage", {})
    model_display_name = extract_model_name(model_usage)
    model_full_name = list(model_usage.keys())[0] if model_usage else "failed-evaluation"
    

    human_minutes = float(metadata.get("T_human", 0))
    

    weight_sub = weights_subfield.get(task_id, {})
    weight_exam = weights_exam.get(task_id, {})
    
    run = {
        "task_id": f"math_olympiad/{task_id}",
        "task_family": "math_olympiad",
        "run_id": f"{model_display_name.lower().replace(' ', '_')}_{task_id}_epoch{epoch}",
        "alias": model_display_name,
        "model": model_full_name,
        "score_cont": score_cont,
        "score_binarized": score_binarized,
        "fatal_error_from": None,
        "human_minutes": human_minutes,
        "human_score": 1.0,  
        "human_source": "baseline",
        "task_source": "Math-Olympiad",
        "generation_cost": 0.0,
        "time_limit": None,
        "started_at": None,
        "completed_at": None,
        "task_version": None,
        "equal_task_weight": weight_sub.get("equal_task_weight", 0),
        "invsqrt_task_weight_subfield": weight_sub.get("invsqrt_task_weight_subfield", 0),
        "invsqrt_task_weight_exam": weight_exam.get("invsqrt_task_weight_exam", 0),
        "subfield": metadata.get("subfield", "Unknown"),
        "exam": metadata.get("exam", "Unknown"),
        "answer_type": metadata.get("answer_type", "Unknown"),
        "classification": metadata.get("classification", "Unknown"),
    }
    
    return [run]


def main() -> None:
    """Main conversion function."""
    parser = argparse.ArgumentParser(
        description="Convert Inspect AI results to METR JSONL format"
    )
    parser.add_argument(
        "--input-file",
        type=pathlib.Path,
        default=pathlib.Path("../results/samples"),
        help="Path to Inspect summaries.json file or samples directory",
    )
    parser.add_argument(
        "--output-file",
        type=pathlib.Path,
        default=pathlib.Path("data/math_olympiad_runs.jsonl"),
        help="Path to output JSONL file",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    
    args = parser.parse_args()
    logging.basicConfig(level=args.log_level)
    
    # Load Inspect results
    samples = load_all_samples(args.input_file)
    
    # Calculate weights for both diversity schemes

    weights_subfield = calculate_task_weights(samples, weight_type="subfield")
    weights_exam = calculate_task_weights(samples, weight_type="exam")
    
    # Log weight distribution
    subfield_dist = defaultdict(int)
    exam_dist = defaultdict(int)
    for sample in samples:
        task_id = sample["id"]
        subfield_dist[weights_subfield[task_id]["category"]] += 1
        exam_dist[weights_exam[task_id]["category"]] += 1

    
    # Convert samples to runs
 
    all_runs = []
    for sample in samples:
        runs = convert_sample_to_runs(sample, weights_subfield, weights_exam)
        all_runs.extend(runs)
    
    # Write output
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_file, "w") as f:
        for run in all_runs:
            f.write(json.dumps(run) + "\n")
    
   


if __name__ == "__main__":
    main()

