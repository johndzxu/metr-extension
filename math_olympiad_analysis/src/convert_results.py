import argparse
import json
import pathlib
from typing import Any


def load_all_samples(input_path: pathlib.Path) -> list[dict[str, Any]]:
    if input_path.is_file():
        with open(input_path) as f:
            return json.load(f)
    elif input_path.is_dir():
        summaries_file = input_path / "summaries.json"
        if summaries_file.exists():
            with open(summaries_file) as f:
                return json.load(f)
        samples = []
        for sample_file in sorted(input_path.glob("*_epoch_*.json")):
            with open(sample_file) as f:
                samples.append(json.load(f))
        return samples
    return []


def find_result_directories(base_path: pathlib.Path) -> list[pathlib.Path]:
    result_dirs = []
    
    for item in base_path.iterdir():
        if item.is_dir() and item.name.startswith("results_"):
            summaries_file = item / "summaries.json"
            if summaries_file.exists():
                result_dirs.append(item)
                print(f"Found result directory: {item.name}")
    
    return sorted(result_dirs)


def load_samples_from_multiple_directories(result_dirs: list[pathlib.Path]) -> list[dict[str, Any]]:
    all_samples = []
    for result_dir in result_dirs:
        summaries_file = result_dir / "summaries.json"
        if summaries_file.exists():
            with open(summaries_file) as f:
                samples = json.load(f)
                all_samples.extend(samples)
                print(f"Loaded {len(samples)} samples from {result_dir.name}")
        else:
            print(f"No summaries.json found in {result_dir.name}")
    
    return all_samples


def convert_score(score_value: str) -> tuple[float, int]:
    if score_value == "C":
        return 1.0, 1
    elif score_value == "I":
        return 0.0, 0
    else:
        print(f"Unknown score value: {score_value}, defaulting to incorrect")
        return 0.0, 0


def extract_model_name(model_usage: dict[str, Any]) -> str:

    if not model_usage:
        return "Failed"  
    
    full_name = list(model_usage.keys())[0]
    
    if full_name.startswith("openrouter/"):
        full_name = full_name.replace("openrouter/", "")
    
    if "/" in full_name:
        parts = full_name.split("/")
        return parts[-1] 
    
    return full_name


def convert_sample_to_runs(sample: dict[str, Any]) -> list[dict[str, Any]]:
    """Convert a single Inspect sample to one or more run entries.
    """

    task_id = sample["id"]
    epoch = sample.get("epoch", 1)
    metadata = sample.get("metadata", {})
    

    scores = sample.get("scores", {})

    score_dict = (
        scores.get("match", {}) or 
        scores.get("latex_exact", {}) or 
        scores.get("olympiadbench_scorer", {}) or
        {}
    )
    score_value = score_dict.get("value", "I")
    

    if not score_dict:
        print(f"No score found for task {task_id}, epoch {epoch}. Available scorers: {list(scores.keys())}")
    
    score_cont, score_binarized = convert_score(score_value)
    

    model_usage = sample.get("model_usage", {})
    model_display_name = extract_model_name(model_usage)
    model_full_name = list(model_usage.keys())[0] if model_usage else "failed-evaluation"
    

    human_minutes = float(metadata.get("T_human", 0))
    
    run = {
        "task_id": f"math_olympiad/{task_id}",
        "task_family": "math_olympiad",
        "run_id": f"{model_display_name.lower().replace(' ', '_')}_{task_id}_epoch{epoch}",
        "alias": model_display_name,
        "model": model_full_name,
        "score_cont": score_cont,
        "score_binarized": score_binarized,
        "human_minutes": human_minutes,
        "human_score": 1.0,  
        "human_source": "baseline",
        "task_source": "Math-Olympiad",
        "subfield": metadata.get("subfield", "Unknown"),
        "exam": metadata.get("exam", "Unknown"),
        "answer_type": metadata.get("answer_type", "Unknown"),
        "classification": metadata.get("classification", "Unknown"),
    }
    
    return [run]


def process_mode(mode: str, base_dir: pathlib.Path, output_dir: pathlib.Path) -> None:
    print(f"\n{'='*60}")
    print(f"Processing {mode.upper()} mode")
    print(f"{'='*60}")
    
    mode_path = base_dir / mode
    
    print(f"Searching for results_* directories in: {mode_path}")
    result_dirs = find_result_directories(mode_path)
    
    
    print(f"Found {len(result_dirs)} result directories")
    samples = load_samples_from_multiple_directories(result_dirs)
    
    
    print(f"Total samples loaded: {len(samples)}")
    
    all_runs = []
    for sample in samples:
        runs = convert_sample_to_runs(sample)
        all_runs.extend(runs)
    
    print(f"Total runs generated: {len(all_runs)}")
    
    output_file = output_dir / f"math_olympiad_runs_{mode}.jsonl"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        for run in all_runs:
            f.write(json.dumps(run) + "\n")



def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert Inspect AI results to JSONL format"
    )
    parser.add_argument(
        "--base-dir",
        type=pathlib.Path,
        default=pathlib.Path(".."),
        help="Base directory containing agentic/non-agentic folders (default: parent directory)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["agentic", "non-agentic", "both"],
        default="both",
        help="Which mode to process: agentic, non-agentic, or both (default: both)",
    )
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=pathlib.Path("data"),
        help="Directory for output JSONL files",
    )
    
    args = parser.parse_args()
    
    base_path = args.base_dir.resolve()
    
    if args.mode in ["agentic", "both"]:
        process_mode("agentic", base_path, args.output_dir)
    
    if args.mode in ["non-agentic", "both"]:
        process_mode("non-agentic", base_path, args.output_dir)
    
   
if __name__ == "__main__":
    main()
