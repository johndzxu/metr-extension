import json
import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.dates as mdates
from scipy.optimize import minimize, curve_fit
from datetime import datetime, timedelta
import yaml
from scipy import stats

def load_release_dates():
    dates_file = pathlib.Path(__file__).parent.parent / "release_dates.yaml"
    with open(dates_file) as f:
        dates = yaml.safe_load(f)
    
    normalized = {}
    for model, date in dates.items():
        key = model.lower().replace('-', '').replace(' ', '').replace(':', '')
        if isinstance(date, str):
            normalized[key] = datetime.strptime(date, '%Y-%m-%d')
        else:
            normalized[key] = datetime.combine(date, datetime.min.time())
    return normalized

def match_model_to_date(model_name, release_dates):
    key = model_name.lower().replace('-', '').replace(' ', '').replace(':', '').replace('free', '')
    
    for date_key, date in release_dates.items():
        if date_key in key or key in date_key:
            return date
    return None

def load_data(jsonl_path):
    data = []
    with open(jsonl_path) as f:
        for line in f:
            data.append(json.loads(line))
    return pd.DataFrame(data)

def fit_logistic_model(df, model_name):
    model_df = df[df['alias'] == model_name].copy()
    
    t_task = model_df['human_minutes'].values
    y = model_df['score_binarized'].values
    
    def negative_log_likelihood(params):
        log_h_model, beta_model = params
        h_model = np.exp(log_h_model)

        logits = (log_h_model - np.log(t_task)) * beta_model
        
        # Sigmoid: p = 1 / (1 + exp(-logits))
        p = 1 / (1 + np.exp(-np.clip(logits, -500, 500)))
        p = np.clip(p, 1e-10, 1 - 1e-10)  
        
        # Negative log-likelihood (no weights)
        nll = -np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))
        return nll
    
    try:
        # Initial guess
        # h_model: start with median task time
        # beta_model: start with 1.0
        log_h_initial = np.log(np.median(t_task))
        beta_initial = 1.0
        
        # Bounds: h_model in [0.1, 10000] minutes, beta_model in [0.01, 10]
        bounds = [
            (np.log(0.1), np.log(10000)),  # log_h_model bounds
            (0.01, 10.0)                    # beta_model bounds (positive)
        ]
        
        result = minimize(
            negative_log_likelihood,
            x0=[log_h_initial, beta_initial],
            method='L-BFGS-B',
            bounds=bounds
        )
        
        log_h_model, beta_model = result.x
        h_model = np.exp(log_h_model)
        
        return h_model, beta_model
        
    except Exception as e:
        print(f" Failed to fit model for {model_name}: {e}")
        return None, None

def logistic_function(t_task, h_model, beta_model):
    logits = (np.log(h_model) - np.log(t_task)) * beta_model
    return 1 / (1 + np.exp(-np.clip(logits, -500, 500)))

def format_time_tick(minutes, pos=None):
    seconds = minutes * 60
    
    if seconds == 0:
        return "0"
    elif seconds < 1:
        return f"{int(seconds*1000)}ms"
    elif seconds < 60:  
        if seconds < 10:
            return f"{seconds:.0f} sec"
        return f"{int(seconds)} sec"
    elif minutes < 60:
        if minutes < 10:
            return f"{minutes:.0f} min"
        return f"{int(minutes)} min"
    elif minutes < 1440:
        hrs = minutes / 60
        if hrs < 10:
            return f"{hrs:.0f} hr{'s' if hrs >= 2 else ''}"
        return f"{int(hrs)} hrs"
    else:
        days = minutes / 1440
        return f"{days:.0f}d"

def exponential_func(x, a, b, c):
    return a * np.exp(b * x) + c

def create_combined_plots(script_dir):
    
    agentic_file = script_dir / "data" / "math_olympiad_runs_agentic.jsonl"
    non_agentic_file = script_dir / "data" / "math_olympiad_runs_non-agentic.jsonl"
    
    
    agentic_df = load_data(agentic_file)
    non_agentic_df = load_data(non_agentic_file)
    
    agentic_df = agentic_df[agentic_df['alias'].notna()].copy()
    agentic_df = agentic_df[agentic_df['alias'] != 'Failed'].copy()
    agentic_df = agentic_df[agentic_df['human_minutes'] > 0].copy()
    
    non_agentic_df = non_agentic_df[non_agentic_df['alias'].notna()].copy()
    non_agentic_df = non_agentic_df[non_agentic_df['alias'] != 'Failed'].copy()
    non_agentic_df = non_agentic_df[non_agentic_df['human_minutes'] > 0].copy()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 8), dpi=100)
    
    release_dates = load_release_dates()
    
    for mode, df, ax in [("agentic", agentic_df, ax1), ("non-agentic", non_agentic_df, ax2)]:
        models = sorted([m for m in df['alias'].unique() if m not in ['Failed', 'Unknown']])
        
        model_dates = []
        for model in models:
            date = match_model_to_date(model, release_dates)
            if date:
                model_dates.append((model, date))
        
        model_dates.sort(key=lambda x: x[1])
        
        min_time = df['human_minutes'].min()
        max_time = df['human_minutes'].max()
        
        color_palette = plt.cm.tab10
        model_colors = {}
        for idx, (model, _) in enumerate(model_dates):
            model_colors[model] = color_palette(idx % 10)
        
        model_fits = {}
        time_horizons_all = []
        
        for model, date in model_dates:
            h_model, beta_model = fit_logistic_model(df, model)
            if h_model is None:
                continue
            
            horizon_50 = h_model
            if horizon_50:
                time_horizons_all.append(horizon_50)
            
            model_fits[model] = {
                'h_model': h_model,
                'beta_model': beta_model,
                'horizon_50': horizon_50,
            }
        
        plot_min_time = min(min_time, min(time_horizons_all) * 0.5) if time_horizons_all else min_time * 0.5
        plot_max_time = max(max_time, max(time_horizons_all) * 2) if time_horizons_all else max_time * 2
        plot_min_time = max(plot_min_time, 0.1)
        
        x_axis_min = plot_min_time * 0.7
        x_axis_max = plot_max_time * 1.1
        
        time_range = np.logspace(np.log10(x_axis_min), np.log10(x_axis_max), 500)
        
        for model, date in model_dates:
            if model not in model_fits:
                continue
            
            fit_data = model_fits[model]
            h_model = fit_data['h_model']
            beta_model = fit_data['beta_model']
            
            color = model_colors[model]
            probabilities = logistic_function(time_range, h_model, beta_model)
            
            ax.plot(time_range, probabilities, linewidth=3, color=color, 
                   label=f"{model} ({date.strftime('%Y-%m')})", zorder=3, alpha=0.9)
        
        ax.axhline(0.5, color='black', linestyle=':', linewidth=2, alpha=0.7, zorder=2,
                   label='50% Success Rate')
        
        ax.set_xscale('log')
        ax.set_xlim(x_axis_min, x_axis_max)
        ax.set_ylim(-0.05, 1.05)
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(format_time_tick))
        
        ax.set_xlabel('Task length (human time)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Success Probability', fontsize=14, fontweight='bold')
        ax.set_title(f'{mode.upper()} models', fontsize=16, fontweight='bold', pad=20)
        
        ax.grid(True, alpha=0.2, which='both', zorder=0, linestyle='-', linewidth=0.5)
        ax.legend(fontsize=9, framealpha=0.95, loc='best', title='Model (Release Date)')
    
    plt.suptitle('Success rate of math olympiad tasks vs task length', 
                 fontsize=18, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    output_file = script_dir.parent / "plots" / "success_rate_vs_task_length.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved combined plot to {output_file}")
    plt.close()

def create_combined_horizon_date_plots(agentic_horizons, non_agentic_horizons, script_dir, log_scale=False):
    suffix = "_log_linear" if log_scale else ""
    plot_title = "log-linear" if log_scale else "exponential"
    
    print(f"CREATING COMBINED TIME HORIZON VS DATE PLOT ({plot_title})")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 8), dpi=100)
    
    for mode, time_horizons, ax in [("agentic", agentic_horizons, ax1), 
                                     ("non-agentic", non_agentic_horizons, ax2)]:
        if not time_horizons:
            continue
        
        dates = []
        horizons = []
        models = []
        
        for model, date, horizon in time_horizons:
            dates.append(date)
            horizons.append(horizon)
            models.append(model)
        
        dates_numeric = [(d - dates[0]).days for d in dates]
        dates_numeric = np.array(dates_numeric, dtype=float)
        horizons_array = np.array(horizons)
        
        if log_scale:
            # Linear fit in log space
            log_horizons = np.log(horizons_array)
            slope, intercept, r_value, p_value, std_err = stats.linregress(dates_numeric, log_horizons)
            r_squared = r_value ** 2
            
            if slope > 0:
                doubling_time_days = np.log(2) / slope
                
            date_range_numeric = np.linspace(dates_numeric.min() - 10, dates_numeric.max() + 30, 200)
            date_range_dates = [dates[0] + timedelta(days=int(d)) for d in date_range_numeric]
            log_fitted_curve = slope * date_range_numeric + intercept
            fitted_curve = np.exp(log_fitted_curve)
            
            label_text = f'Linear fit (R² = {r_squared:.3f})\nDoubling time: {doubling_time_days:.0f} days'
            ax.plot(date_range_dates, fitted_curve, '--', color='gray', linewidth=2, 
                   alpha=0.6, zorder=2, label=label_text)
            
            ax.set_yscale('log')
        else:
            # Exponential fit
            initial_guess = [1.0, 0.01, 0.0]
            popt, _ = curve_fit(exponential_func, dates_numeric, horizons_array, 
                               p0=initial_guess, maxfev=5000)
            a, b, c = popt
            
            fitted_values = exponential_func(dates_numeric, a, b, c)
            residuals = horizons_array - fitted_values
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((horizons_array - np.mean(horizons_array))**2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            date_range_numeric = np.linspace(dates_numeric.min() - 10, dates_numeric.max() + 30, 200)
            date_range_dates = [dates[0] + timedelta(days=int(d)) for d in date_range_numeric]
            fitted_curve = exponential_func(date_range_numeric, a, b, c)
            
            label_text = f'Exponential fit (R² = {r_squared:.3f})'
            ax.plot(date_range_dates, fitted_curve, '--', color='gray', linewidth=2, 
                   alpha=0.6, zorder=2, label=label_text)
        
        colors = plt.cm.tab10
        for idx, (model, date, horizon) in enumerate(time_horizons):
            color = colors(idx % 10)
            ax.scatter(date, horizon, s=25, color=color, alpha=0.75, zorder=3, 
                      edgecolors='black', linewidth=0.6)
            ax.annotate(model, (date, horizon), xytext=(10, 10), textcoords='offset points',
                       fontsize=9, alpha=0.8, bbox=dict(boxstyle='round,pad=0.3', 
                       facecolor='white', alpha=0.7, edgecolor='gray'))
        
        ax.set_xlabel('LLM Release Date', fontsize=14, fontweight='bold')
        ax.set_ylabel('Task Duration (for humans)', fontsize=14, fontweight='bold')
        ax.set_title(f'{mode.upper()} models', fontsize=16, fontweight='bold', pad=20)
        
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: format_time_tick(x)))
        
        if log_scale:
            y_ticks = [0.1, 0.3, 1, 3, 10, 30, 100, 300, 600]
            ax.set_yticks(y_ticks)
            ax.set_ylim(0.1, 600)
        else:
            y_ticks = [0, 60, 120, 180, 240, 300, 360, 420, 480, 540, 600]
            ax.set_yticks(y_ticks)
            ax.set_ylim(-10, 600)
        
        ax.grid(True, alpha=0.3, zorder=0, linestyle='-', linewidth=0.5, which='both')
        ax.legend(fontsize=10, framealpha=0.9, loc='best')
    
    plt.suptitle('Time horizon of math olympiad problems models can complete 50% of the time', 
                 fontsize=18, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    output_file = script_dir.parent / "plots" / f"time_horizon_vs_release_date{suffix}.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()

def extract_time_horizons(mode, script_dir):
    data_file = script_dir / "data" / f"math_olympiad_runs_{mode}.jsonl"
    
    df = load_data(data_file)
    df = df[df['alias'].notna()].copy()
    df = df[df['alias'] != 'Failed'].copy()
    df = df[df['human_minutes'] > 0].copy()
    
    models = sorted([m for m in df['alias'].unique() if m not in ['Failed', 'Unknown']])
    release_dates = load_release_dates()
    
    model_dates = []
    for model in models:
        date = match_model_to_date(model, release_dates)
        if date:
            model_dates.append((model, date))
    
    if not model_dates:
        return None
    
    model_dates.sort(key=lambda x: x[1])
    
    time_horizons = []
    
    for model, date in model_dates:
        h_model, beta_model = fit_logistic_model(df, model)
        
        if h_model is None:
            continue
        
        horizon_50 = h_model
        if horizon_50:
            time_horizons.append((model, date, horizon_50))
    
    return time_horizons

def main():
    script_dir = pathlib.Path(__file__).parent
    
    agentic_horizons = extract_time_horizons("agentic", script_dir)
    non_agentic_horizons = extract_time_horizons("non-agentic", script_dir)
    
    create_combined_plots(script_dir)
    
    if agentic_horizons and non_agentic_horizons:
        create_combined_horizon_date_plots(agentic_horizons, non_agentic_horizons, script_dir, log_scale=False)
        create_combined_horizon_date_plots(agentic_horizons, non_agentic_horizons, script_dir, log_scale=True)
    

if __name__ == '__main__':
    main()
