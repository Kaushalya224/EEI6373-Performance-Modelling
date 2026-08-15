"""
EEI6373 - Performance Modelling Mini Project
Supermarket Checkout Queue & Multi-Counter Analysis Pipeline
"""

import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------------------------------------------------------
# 1. Dataset Generation / Loading Engine
# ---------------------------------------------------------
def load_or_generate_dataset(csv_path="dataset_3.csv"):
  """Loads dataset_3.csv or generates synthetic peak-hour data matching schema:

  - timestamp: ISO-8601 recording time (15-min intervals, 5:00 PM - 9:00 PM)
  - arrival_rate_lambda: Incoming customers per hour (cust/hr)
  - open_counters_c: Number of active checkout counters
  - service_rate_mu: Average cashier processing rate (cust/hr)
  - avg_queue_length_Lq: Average customers waiting in line
  - avg_wait_time_min: Average queue wait time before service (minutes)
  - cashier_utilization_pct: Average cashier busy percentage
  - balked_customers: Count of customers leaving due to excessive lines
  """
  try:
    df = pd.read_csv(csv_path)
    print(f"Successfully loaded '{csv_path}' with {len(df)} records.")
    return df
  except FileNotFoundError:
    print(f"File '{csv_path}' not found. Generating synthetic peak dataset...")

  # 15-minute intervals from 5:00 PM (17:00) to 9:00 PM (21:00)
  timestamps = pd.date_range("2026-03-06 17:00", "2026-03-06 20:45", freq="15min")

  # Peak arrival rates (cust/hr) and dynamic server allocations
  lambdas = [45, 52, 65, 78, 92, 105, 115, 110, 98, 85, 72, 60, 50, 42, 38, 35]
  counters = [2, 2, 3, 3, 3, 4, 4, 4, 4, 3, 3, 2, 2, 2, 2, 2]
  mu = 30.0  # Mean service rate: 30 cust/hr per cashier (2.0 min service time)

  data = []
  for ts, lam, c in zip(timestamps, lambdas, counters):
    rho = lam / (c * mu)
    a = lam / mu

    if rho < 1.0:
      # Standard M/M/c Queue Calculations
      sum_k = sum([(a**k) / math.factorial(k) for k in range(c)])
      last_term = (a**c) / (math.factorial(c) * (1.0 - rho))
      p0 = 1.0 / (sum_k + last_term)
      erlang_c = last_term * p0
      Lq = (erlang_c * rho) / (1.0 - rho)
      Wq_min = (Lq / lam) * 60.0
      balked = 0
    else:
      # Congested Queue State (rho >= 1.0)
      Lq = (lam - c * mu) * 0.25 + 8.5
      Wq_min = (Lq / lam) * 60.0
      balked = int((rho - 0.95) * 10)

    util_pct = min(rho * 100.0, 98.5)

    data.append({
        "timestamp": ts.strftime("%Y-%m-%dT%H:%M:%S"),
        "arrival_rate_lambda": lam,
        "open_counters_c": c,
        "service_rate_mu": mu,
        "avg_queue_length_Lq": round(Lq, 2),
        "avg_wait_time_min": round(Wq_min, 2),
        "cashier_utilization_pct": round(util_pct, 1),
        "balked_customers": balked,
    })

  df = pd.DataFrame(data)
  df.to_csv(csv_path, index=False)
  print(f"Saved generated dataset to '{csv_path}'.")
  return df


# ---------------------------------------------------------
# 2. Performance Audit and Target KPI Evaluator
# ---------------------------------------------------------
def evaluate_system_performance(df):
  """Evaluates data against defined EEI6373 performance objectives:

  1. Average Queue Wait Time (Target: Wq < 3.0 min)
  2. Cashier Utilization (Target: 70% <= rho <= 85%)
  3. Total Customer Balking & Peak Bottlenecks
  """
  print("\n================ SYSTEM PERFORMANCE AUDIT ================")
  print(f"Total Recording Intervals : {len(df)} (15-minute blocks)")
  print(
      f"Peak Arrival Rate (lambda): {df['arrival_rate_lambda'].max()} cust/hr"
  )
  print(f"Mean Arrival Rate         : {df['arrival_rate_lambda'].mean():.2f}"
        " cust/hr")
  print(
      f"Mean Queue Wait Time (Wq) : {df['avg_wait_time_min'].mean():.2f} min"
  )
  print(
      f"Max Queue Wait Time       : {df['avg_wait_time_min'].max():.2f} min"
  )
  print(
      "Mean Cashier Utilization  :"
      f" {df['cashier_utilization_pct'].mean():.2f}%"
  )
  print(f"Total Balked Customers    : {df['balked_customers'].sum()}")

  # KPI Compliance Checks
  wait_kpi_pass = df["avg_wait_time_min"] < 3.0
  util_kpi_pass = (df["cashier_utilization_pct"] >= 70.0) & (
      df["cashier_utilization_pct"] <= 85.0
  )

  print("\n---------------- KPI COMPLIANCE SUMMARY ----------------")
  print(
      f"Wait Time Target (<3 min) Compliance : {wait_kpi_pass.mean()*100:.1f}%"
      " of intervals"
  )
  print(
      "Utilization Target (70-85%) Compliance:"
      f" {util_kpi_pass.mean()*100:.1f}% of intervals"
  )
  print(
      "Over-utilized Intervals (>85%)       :"
      f" {(df['cashier_utilization_pct'] > 85.0).sum()} intervals"
  )
  print(
      "Under-utilized Intervals (<70%)      :"
      f" {(df['cashier_utilization_pct'] < 70.0).sum()} intervals"
  )


# ---------------------------------------------------------
# 3. Dynamic Counter Optimization Engine
# ---------------------------------------------------------
def calculate_optimal_counters(
    lam, mu, target_util_min=0.70, target_util_max=0.85
):
  """Calculates recommended open registers c for an arrival rate lam."""
  # c_min guarantees stability (rho < 1.0)
  c_min = int(math.ceil(lam / mu))
  for c in range(max(1, c_min), c_min + 5):
    rho = lam / (c * mu)
    if rho <= target_util_max:
      return c
  return max(1, c_min)


# ---------------------------------------------------------
# 4. Visualization & Plotting Pipeline
# ---------------------------------------------------------
def generate_performance_plots(df, output_img="queue_performance.png"):
  """Generates publication-quality dual-axis queue performance charts."""
  fig, ax1 = plt.subplots(figsize=(12, 6))

  times = [t[11:16] for t in df["timestamp"]]  # Extract HH:MM

  # Axis 1: Wait Time and Queue Length
  color_wait = "tab:red"
  ax1.set_xlabel("Time Block (Friday Peak 5:00 PM - 9:00 PM)", fontweight="bold")
  ax1.set_ylabel("Average Wait Time (min)", color=color_wait, fontweight="bold")
  (line1,) = ax1.plot(
      times,
      df["avg_wait_time_min"],
      color=color_wait,
      marker="o",
      linewidth=2,
      label="Avg Wait Time Wq (min)",
  )
  ax1.axhline(
      y=3.0,
      color="red",
      linestyle="--",
      alpha=0.7,
      label="Target Max Wait (3 min)",
  )
  ax1.tick_params(axis="y", labelcolor=color_wait)

  # Axis 2: Cashier Utilization %
  ax2 = ax1.twinx()
  color_util = "tab:blue"
  ax2.set_ylabel(
      "Cashier Utilization (%)", color=color_util, fontweight="bold"
  )
  (line2,) = ax2.plot(
      times,
      df["cashier_utilization_pct"],
      color=color_util,
      marker="s",
      linestyle="-.",
      linewidth=2,
      label="Utilization (%)",
  )
  ax2.axhspan(
      70.0,
      85.0,
      color="blue",
      alpha=0.1,
      label="Optimal Utilization Zone (70-85%)",
  )
  ax2.tick_params(axis="y", labelcolor=color_util)

  plt.title(
      "Supermarket Multi-Counter System Performance Analysis (EEI6373)",
      fontsize=14,
      fontweight="bold",
  )
  ax1.set_xticks(range(len(times)))
  ax1.set_xticklabels(times, rotation=45)
  ax1.grid(True, linestyle=":", alpha=0.6)

  # Combine legends
  lines1, labels1 = ax1.get_legend_handles_labels()
  lines2, labels2 = ax2.get_legend_handles_labels()
  ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

  fig.tight_layout()
  plt.savefig(output_img, dpi=300)
  print(f"\nVisualization saved to '{output_img}'.")
  plt.close()


# ---------------------------------------------------------
# 5. Main Execution Entry Point
# ---------------------------------------------------------
if __name__ == "__main__":
  # 1. Load or create dataset matching schema
  df = load_or_generate_dataset("dataset_3.csv")

  # 2. Add optimal cashier recommendation column
  df["recommended_counters_c"] = df["arrival_rate_lambda"].apply(
      lambda lam: calculate_optimal_counters(lam, mu=30.0)
  )

  # 3. Print sample data records
  print("\n--- DATASET SAMPLE (dataset_3.csv Schema) ---")
  print(
      df[[
          "timestamp",
          "arrival_rate_lambda",
          "open_counters_c",
          "recommended_counters_c",
          "avg_wait_time_min",
          "cashier_utilization_pct",
          "balked_customers",
      ]].head(8)
  )

  # 4. Perform full performance evaluation
  evaluate_system_performance(df)

  # 5. Output visual plot
  generate_performance_plots(df)
