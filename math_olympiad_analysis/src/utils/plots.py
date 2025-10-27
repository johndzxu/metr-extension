import logging
import pathlib


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure
from scipy import stats

logger = logging.getLogger(__name__)


class MathOlympiadPlotter:
    
    def __init__(
        self,
        figsize: tuple[int, int] = (10, 6),
        font_family: str = 'sans-serif'
    ):
        self.figsize = figsize
        self.font_family = font_family
        

        plt.style.use('seaborn-v0_8-darkgrid')

        sns.set_context("notebook", font_scale=1.1)
        
        self.model_colors = {
            'DeepSeek-R1': '#2E7D32',          # Green
            'minimax-m2:free': '#1976D2',      # Blue
            'Qwen2.5-Coder-32B-Instruct': '#F57C00',  # Orange
        }
        
        self.subfield_colors = {
            'Algebra': '#E53935',       # Red
            'Geometry': '#1E88E5',      # Blue  
            'Combinatorics': '#43A047', # Green
            'Number Theory': '#FB8C00', # Orange
        }
        
    def plot_task_difficulty_distribution(
        self,
        df: pd.DataFrame,
        time_col: str = 'human_minutes',
        title: str = "Task Difficulty Distribution"
    ) -> Figure:
      
        fig, ax = plt.subplots(figsize=self.figsize, dpi=100)
        
       
        times = df[time_col].values
        times = times[times > 0]  
        

        log_bins = np.logspace(np.log10(max(times.min(), 0.1)), 
                               np.log10(times.max()), 20)
        
        ax.hist(times, bins=log_bins, color='#1976D2', 
                alpha=0.7, edgecolor='black', linewidth=1)
        

        ax.set_xlabel('Human Completion Time (minutes)', fontsize=12)
        ax.set_ylabel('Number of Tasks', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        ax.set_xscale('log')
        ax.grid(True, alpha=0.3, which='both')
        

        median_time = np.median(times)
        ax.axvline(median_time, color='red', linestyle='--', 
                  linewidth=2, alpha=0.7, label=f'Median: {median_time:.1f} min')
        ax.legend(fontsize=10, framealpha=0.9)
        

        plt.tight_layout()
        
        return fig
    
    def plot_success_vs_difficulty(
        self,
        df: pd.DataFrame,
        time_col: str = 'human_minutes',
        success_col: str = 'score_binarized',
        model_col: str = 'alias',
        title: str = "Model Success Rate vs Task Difficulty",
        show_trend: bool = True,
        log_scale: bool = True
    ) -> Figure:

        fig, ax = plt.subplots(figsize=(12, 7), dpi=100)
        
        df_filtered = df[df[time_col] > 0].copy()
        
        models = sorted([m for m in df_filtered[model_col].unique() 
                        if m != 'Failed' and m != 'Unknown'])
        
        # Plot each model
        for model in models:
            model_df = df_filtered[df_filtered[model_col] == model]
            
            # Aggregate by task
            task_stats = model_df.groupby('task_id').agg({
                time_col: 'first',
                success_col: 'mean'
            }).reset_index()
            
            # Get color
            color = self.model_colors.get(model, '#757575')
            
            # Plot with larger markers
            ax.scatter(task_stats[time_col], task_stats[success_col], 
                      label=model, alpha=0.7, s=100, color=color, 
                      edgecolors='black', linewidth=0.5)
        
        # Add trend line for all data
        if show_trend and len(df_filtered) > 5:
            task_stats_all = df_filtered.groupby('task_id').agg({
                time_col: 'first',
                success_col: 'mean'
            }).reset_index()
            
            times = task_stats_all[time_col].values
            success = task_stats_all[success_col].values
            
            if len(np.unique(times)) > 1:
                if log_scale:
                    log_times = np.log10(times)
                    slope, intercept, r_value, _, _ = stats.linregress(log_times, success)
                    
                    x_trend = np.logspace(np.log10(times.min()), 
                                         np.log10(times.max()), 100)
                    y_trend = slope * np.log10(x_trend) + intercept
                    
                    ax.plot(x_trend, y_trend, 'k--', alpha=0.5, linewidth=2,
                           label=f'Trend (R²={r_value**2:.2f})')
        

        ax.set_xlabel('Human Completion Time (minutes)', fontsize=12)
        ax.set_ylabel('Model Success Rate', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        
        if log_scale:
            ax.set_xscale('log')
            
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3, which='both')
        ax.legend(fontsize=10, framealpha=0.9, loc='best')
        
        plt.tight_layout()
        
        return fig
    
    def plot_subfield_performance(
        self,
        df: pd.DataFrame,
        subfield_col: str = 'subfield',
        success_col: str = 'score_binarized',
        model_col: str = 'alias',
        title: str = "Model Performance by Subfield"
    ) -> Figure:
        """Plot clean bar chart of performance by subfield."""
        fig, ax = plt.subplots(figsize=(12, 6), dpi=100)
        
        # Filter out failed and unknown
        df_filtered = df[(df[model_col] != 'Failed') & 
                         (df[subfield_col] != 'Unknown')].copy()
        

        models = sorted(df_filtered[model_col].unique())
        subfields = sorted(df_filtered[subfield_col].unique())
        
        # Calculate success rates
        data_matrix = []
        for model in models:
            model_data = []
            for subfield in subfields:
                subset = df_filtered[(df_filtered[model_col] == model) & 
                                    (df_filtered[subfield_col] == subfield)]
                if len(subset) > 0:
                    success_rate = subset[success_col].mean()
                    model_data.append(success_rate * 100)
                else:
                    model_data.append(0)
            data_matrix.append(model_data)
        
        # Create grouped bar chart
        x = np.arange(len(subfields))
        width = 0.8 / len(models)
        
        for i, (model, data) in enumerate(zip(models, data_matrix)):
            offset = (i - len(models)/2 + 0.5) * width
            color = self.model_colors.get(model, '#757575')
            bars = ax.bar(x + offset, data, width, label=model, 
                         color=color, alpha=0.8, edgecolor='black', linewidth=0.5)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                if height > 5:  
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.0f}%',
                           ha='center', va='bottom', fontsize=8)
        
        ax.set_xlabel('Mathematical Subfield', fontsize=12)
        ax.set_ylabel('Success Rate (%)', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(subfields, rotation=0)
        ax.set_ylim(0, 105)
        ax.grid(True, alpha=0.3, axis='y')
        ax.legend(fontsize=10, framealpha=0.9, loc='upper right')
        
        plt.tight_layout()
        
        return fig
    
    def save_figure(self, fig: Figure, path: pathlib.Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
