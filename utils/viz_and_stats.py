import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import ttest_rel

def create_plots_and_stats(results, functions, algorithms, complexity_data):
    summary_data = []
    for func_name in functions:
        row = {"Function": func_name}
        for algo_name in algorithms:
            fitnesses = results[algo_name][func_name]["fitness"]
            times = results[algo_name][func_name]["time"]
            row.update({
                f"{algo_name} Mean Fitness": np.mean(fitnesses),
                f"{algo_name} Std Fitness": np.std(fitnesses),
                f"{algo_name} Mean Time (s)": np.mean(times)
            })
        
        # Perform t-tests against a baseline algorithm if it exists
        base_algo = "GA" 
        if base_algo in results and base_algo in results[list(results.keys())[0]]:
            base_fitnesses = results[base_algo][func_name]["fitness"]
            for algo_name in algorithms:
                if algo_name != base_algo and algo_name in results and func_name in results[algo_name]:
                    other_fitnesses = results[algo_name][func_name]["fitness"]
                    try:
                        t_stat, p_value = ttest_rel(base_fitnesses, other_fitnesses)
                        significant = p_value < 0.05
                    except ValueError:
                        p_value = 1.0
                        significant = False
                    row.update({
                        f"P-Value ({base_algo} vs {algo_name})": p_value,
                        f"Significant ({base_algo} vs {algo_name})": significant
                    })
        
        summary_data.append(row)

        # Plot convergence for each function
        fig = go.Figure()
        for algo_name in algorithms:
            if algo_name not in ["DE", "PSO"]:
                if algo_name in results and func_name in results[algo_name]:
                    logbooks = results[algo_name][func_name]["logbook"]
                    if logbooks:
                        min_fitness_runs = []
                        max_len = 0
                        for logbook in logbooks:
                            if logbook:
                                min_fitness_runs.append(logbook.select("min"))
                                max_len = max(max_len, len(logbook.select("gen")))
                        
                        if min_fitness_runs:
                            avg_fitness = np.full(max_len, np.nan)
                            for i in range(max_len):
                                vals = [run[i] for run in min_fitness_runs if i < len(run)]
                                if vals:
                                    avg_fitness[i] = np.mean(vals)

                            gens = list(range(max_len))
                            fig.add_trace(go.Scatter(x=gens, y=avg_fitness, mode="lines", name=algo_name))

        fig.update_layout(
            title=f"<b>Average Convergence Plot for {func_name}</b>",
            xaxis_title="Generation",
            yaxis_title="Average Best Fitness (Lower is Better)",
            template="plotly_white",
            legend_title="Algorithm"
        )
        # fig.show()
        fig.write_html(f"convergence_{func_name}.html")


    # Create and print summary DataFrame
    results_df = pd.DataFrame(summary_data)
    print("\n--- Results DataFrame ---")
    pd.set_option('display.float_format', lambda x: '%.3e' % x)
    print(results_df)

    # Save summary to CSV
    results_df.to_csv("optimization_results.csv", index=False)

    # Fitness Bar Chart
    fig_fitness = go.Figure()
    for algo_name in algorithms:
        fig_fitness.add_trace(go.Bar(
            name=algo_name,
            x=results_df["Function"],
            y=results_df[f"{algo_name} Mean Fitness"],
            error_y=dict(type="data", array=results_df[f"{algo_name} Std Fitness"])
        ))
    fig_fitness.update_layout(
        barmode="group",
        title_text="<b>Mean Fitness Comparison Across Algorithms</b>",
        xaxis_title="Benchmark Function",
        yaxis_title="Mean Best Fitness (Log Scale)",
        yaxis_type="log",
        legend_title="Algorithm",
        template="plotly_white"
    )
    # fig_fitness.show()
    fig_fitness.write_html("fitness_comparison.html")

    # Time Bar Chart
    fig_time = go.Figure()
    for algo_name in algorithms:
        fig_time.add_trace(go.Bar(
            name=algo_name,
            x=results_df["Function"],
            y=results_df[f"{algo_name} Mean Time (s)"]
        ))
    fig_time.update_layout(
        barmode="group",
        title_text="<b>Mean Execution Time Comparison</b>",
        xaxis_title="Benchmark Function",
        yaxis_title="Mean Execution Time (seconds)",
        legend_title="Algorithm",
        template="plotly_white"
    )
    # fig_time.show()
    fig_time.write_html("time_comparison.html")
