#Analyze math olympiad evaluation results.

import argparse
import json
import logging
import pathlib
import matplotlib.pyplot as plt
import pandas as pd


from utils.plots import MathOlympiadPlotter
from utils.logistic import fit_math_olympiad_model

logger = logging.getLogger(__name__)


def load_runs(jsonl_path: pathlib.Path) -> pd.DataFrame:
    
    runs = []
    with open(jsonl_path) as f:
        for line in f:
            runs.append(json.loads(line))
    return pd.DataFrame(runs)


def plot_task_distribution(
    df: pd.DataFrame,
    output_dir: pathlib.Path,
    weight_col: str = "equal_task_weight"
) -> None:
    

    tasks_df = df.groupby("task_id").first()
    

    plotter = MathOlympiadPlotter()
    fig = plotter.plot_task_difficulty_distribution(
        tasks_df,
        time_col="human_minutes",
        title=f"Distribution of Task Difficulties ({weight_col or 'unweighted'})"
    )
    
    output_path = output_dir / f"task_distribution_{weight_col or 'unweighted'}.png"
    plotter.save_figure(fig, output_path)
    plt.close(fig)


def plot_success_by_difficulty(
    df: pd.DataFrame,
    output_dir: pathlib.Path,
    weight_col: str = "equal_task_weight"
) -> None:
    
    plotter = MathOlympiadPlotter(figsize=(12, 7))
    fig = plotter.plot_success_vs_difficulty(
        df,
        time_col="human_minutes",
        success_col="score_binarized",
        model_col="alias",
        title=f"Model Success Rate vs Task Difficulty ({weight_col or 'unweighted'})"
    )
    
    output_path = output_dir / f"success_by_difficulty_{weight_col or 'unweighted'}.png"
    plotter.save_figure(fig, output_path)
    plt.close(fig)


def plot_subfield_performance(
    df: pd.DataFrame,
    output_dir: pathlib.Path
) -> None:
    
    plotter = MathOlympiadPlotter(figsize=(12, 6))
    fig = plotter.plot_subfield_performance(
        df,
        subfield_col="subfield",
        success_col="score_binarized",
        model_col="alias"
    )
    
    output_path = output_dir / "performance_by_subfield.png"
    plotter.save_figure(fig, output_path)
    plt.close(fig)


def analyze_time_horizons(df: pd.DataFrame, output_dir: pathlib.Path) -> None:
    """Analyze time horizons using logistic regression.
    
    Args:
        df: DataFrame with run data
        output_dir: Directory to save analysis
    """

    # Fit models for each weighting scheme
    weighting_schemes = ["equal_task_weight", "invsqrt_task_weight_subfield", "invsqrt_task_weight_exam"]
    horizon_results = {}
    
    for weight_col in weighting_schemes:
        if weight_col not in df.columns:
            continue
            
        try:

            summary = fit_math_olympiad_model(
                df,
                time_col="human_minutes",
                success_col="score_binarized",
                weight_col=weight_col,
                log_transform=True,
                regularization=0.01,
                cv_folds=5
            )
            
            horizon_results[weight_col] = summary
            
            logger.info(f"Time horizons for {weight_col}:")
            for rate in [25, 50, 75, 90]:
                horizon_key = f'time_horizon_{rate}pct'
                if horizon_key in summary and summary[horizon_key] is not None:
                    horizon_minutes = summary[horizon_key]
                    logger.info(f"  {rate}% horizon: {horizon_minutes:.1f} minutes")
                    
        except Exception as e:
            logger.warning(f"Failed to fit model for {weight_col}: {e}")
            continue
    

    horizon_file = output_dir / "time_horizon_analysis.json"
    with open(horizon_file, 'w') as f:
        json.dump(horizon_results, f, indent=2)
    



def generate_summary_statistics(df: pd.DataFrame, output_dir: pathlib.Path) -> None:
   
    summary = {}
    

    summary["total_runs"] = len(df)
    summary["unique_tasks"] = df["task_id"].nunique()
    summary["models"] = df["alias"].unique().tolist()
    
    # Per-model statistics
    model_stats = {}
    for model in df["alias"].unique():
        model_df = df[df["alias"] == model]
        model_stats[model] = {
            "num_runs": len(model_df),
            "num_tasks": model_df["task_id"].nunique(),
            "mean_score": float(model_df["score_cont"].mean()),
            "success_rate": float(model_df["score_binarized"].mean()),
            "median_human_time": float(model_df.groupby("task_id")["human_minutes"].first().median())
        }
    summary["model_statistics"] = model_stats
    
    # Subfield statistics
    subfield_stats = {}
    for subfield in df["subfield"].unique():
        if subfield != "Unknown":
            subfield_df = df[df["subfield"] == subfield]
            subfield_stats[subfield] = {
                "num_tasks": subfield_df["task_id"].nunique(),
                "mean_success_rate": float(subfield_df["score_binarized"].mean()),
                "mean_human_time": float(subfield_df.groupby("task_id")["human_minutes"].first().mean())
            }
    summary["subfield_statistics"] = subfield_stats
    
    # Save summary
    output_path = output_dir / "summary_statistics.json"
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)

    

    logger.info(f"Total runs: {summary['total_runs']}")
    logger.info(f"Unique tasks: {summary['unique_tasks']}")
    logger.info(f"Models: {', '.join(summary['models'])}")
    logger.info("\nPer-model statistics:")
    for model, stats in model_stats.items():
        logger.info(f"  {model}:")
        logger.info(f"    Success rate: {stats['success_rate']:.1%}")
        logger.info(f"    Tasks evaluated: {stats['num_tasks']}")


def main():

    parser = argparse.ArgumentParser(
        description="Analyze math olympiad evaluation results"
    )
    parser.add_argument(
        "--runs-file",
        type=pathlib.Path,
        default=pathlib.Path("data/math_olympiad_runs.jsonl"),
        help="Path to runs JSONL file",
    )
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=pathlib.Path("plots"),
        help="Directory to save plots and analysis",
    )
    parser.add_argument(
        "--weight-col",
        type=str,
        default="invsqrt_task_weight_subfield",
        choices=["equal_task_weight", "invsqrt_task_weight_subfield", "invsqrt_task_weight_exam", None],
        help="Column to use for task weighting",
    )
    
    args = parser.parse_args()
    logging.basicConfig(level="INFO")
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    

    df = load_runs(args.runs_file)

    
    # Filter out failed evaluations
    df_original = df.copy()
    df = df[df["alias"] != "Failed"].copy()
    if len(df_original) > len(df):
        logger.info(f"Filtered out {len(df_original) - len(df)} failed evaluation runs")
        logger.info(f"Working with {len(df)} runs for {df['task_id'].nunique()} tasks")
    

    plot_task_distribution(df, args.output_dir, args.weight_col)
    plot_success_by_difficulty(df, args.output_dir, args.weight_col)
    plot_subfield_performance(df, args.output_dir)
    
    analyze_time_horizons(df, args.output_dir)
    

    generate_summary_statistics(df, args.output_dir)
    



if __name__ == "__main__":
    main()

