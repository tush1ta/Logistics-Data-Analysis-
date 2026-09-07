"""
Week 3 Task: Advanced Data Analysis and Visualization in Logistics
====================================================================
This script simulates a realistic logistics operations dataset, performs
exploratory data analysis (EDA), and generates a series of visualizations
that support operational, cost, and performance insights.

Author: Data Analysis Task
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ----------------------------------------------------------------------
# 0. Global settings
# ----------------------------------------------------------------------
np.random.seed(42)
sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams["figure.dpi"] = 150
OUT = "images/"

# ----------------------------------------------------------------------
# 1. DATA SIMULATION
# ----------------------------------------------------------------------
# We simulate one year (2025) of shipment-level records for a mid-size
# logistics operator running deliveries across five regional hubs and
# four transportation modes.
#
# Key variables:
#   shipment_id        - unique identifier
#   date                - shipment date (daily granularity across FY2025)
#   region              - delivery region / distribution hub
#   transport_mode      - Road, Rail, Air, Sea
#   shipment_volume_kg  - weight/volume of the shipment
#   distance_km         - distance travelled
#   delivery_time_hrs   - actual time from dispatch to delivery
#   promised_time_hrs   - contractual/SLA delivery time
#   transportation_cost - total cost of moving the shipment (USD)
#   fuel_cost           - fuel component of transportation cost (USD)
#   labor_cost          - labor component of transportation cost (USD)
#   delay_flag          - 1 if delivered later than promised_time_hrs
#   customer_rating     - post-delivery satisfaction score (1-5)

n = 5000
regions = ["North", "South", "East", "West", "Central"]
region_weights = [0.24, 0.20, 0.18, 0.22, 0.16]
modes = ["Road", "Rail", "Air", "Sea"]
mode_weights = [0.55, 0.20, 0.15, 0.10]

# Mode-specific base parameters (speed profile & cost profile)
mode_params = {
    "Road": {"speed_kmh": 55,  "cost_per_km": 1.10, "base_cost": 40,  "variability": 0.20},
    "Rail": {"speed_kmh": 65,  "cost_per_km": 0.65, "base_cost": 90,  "variability": 0.15},
    "Air":  {"speed_kmh": 700, "cost_per_km": 3.50, "base_cost": 250, "variability": 0.30},
    "Sea":  {"speed_kmh": 35,  "cost_per_km": 0.30, "base_cost": 150, "variability": 0.25},
}

dates = pd.date_range("2025-01-01", "2025-12-31", freq="D")

df = pd.DataFrame({
    "shipment_id": [f"SHP{100000+i}" for i in range(n)],
    "date": np.random.choice(dates, size=n),
    "region": np.random.choice(regions, size=n, p=region_weights),
    "transport_mode": np.random.choice(modes, size=n, p=mode_weights),
})

# Distance depends loosely on mode (air/sea legs tend to be longer haul)
distance_base = df["transport_mode"].map({
    "Road": 250, "Rail": 500, "Air": 1200, "Sea": 2000
})
df["distance_km"] = np.round(
    np.random.gamma(shape=2.2, scale=distance_base / 2.2), 1
)

# Shipment volume (kg) - right-skewed, typical of mixed freight
df["shipment_volume_kg"] = np.round(np.random.lognormal(mean=6.2, sigma=0.9, size=n), 1)
df["shipment_volume_kg"] = df["shipment_volume_kg"].clip(20, 20000)

# Delivery time: base on distance/speed + mode variability + regional congestion factor
region_congestion = {"North": 1.00, "South": 1.08, "East": 1.15, "West": 0.95, "Central": 1.20}

def simulate_delivery_time(row):
    params = mode_params[row["transport_mode"]]
    base_hours = row["distance_km"] / params["speed_kmh"]
    congestion = region_congestion[row["region"]]
    noise = np.random.normal(1.0, params["variability"])
    handling_hours = np.random.uniform(2, 12)  # loading/unloading/customs etc.
    return max(base_hours * congestion * noise + handling_hours, 1)

df["delivery_time_hrs"] = df.apply(simulate_delivery_time, axis=1).round(1)

# Promised (SLA) time: typically set with a buffer over an "ideal" delivery time
df["promised_time_hrs"] = (
    (df["distance_km"] / df["transport_mode"].map({m: mode_params[m]["speed_kmh"] for m in modes}))
    * 1.25 + 8
).round(1)

df["delay_hrs"] = (df["delivery_time_hrs"] - df["promised_time_hrs"]).round(1)
df["delay_flag"] = (df["delay_hrs"] > 0).astype(int)

# Transportation cost = base + per-km cost (scaled by mode) + volume surcharge + noise
def simulate_cost(row):
    params = mode_params[row["transport_mode"]]
    distance_cost = row["distance_km"] * params["cost_per_km"]
    volume_surcharge = row["shipment_volume_kg"] * 0.015
    noise = np.random.normal(1.0, 0.12)
    return max((params["base_cost"] + distance_cost + volume_surcharge) * noise, 10)

df["transportation_cost"] = simulate_cost_series = df.apply(simulate_cost, axis=1).round(2)

# Split cost into fuel vs labor components (illustrative ratios by mode)
fuel_ratio = df["transport_mode"].map({"Road": 0.45, "Rail": 0.35, "Air": 0.60, "Sea": 0.55})
df["fuel_cost"] = (df["transportation_cost"] * fuel_ratio).round(2)
df["labor_cost"] = (df["transportation_cost"] - df["fuel_cost"]).round(2)

# Customer satisfaction: negatively correlated with delay, with noise
df["customer_rating"] = (
    5 - (df["delay_flag"] * np.random.uniform(0.5, 2.0, size=n))
    - (df["delay_hrs"].clip(lower=0) / 20)
    + np.random.normal(0, 0.4, size=n)
).clip(1, 5).round(1)

df["month"] = df["date"].dt.month_name().str[:3]
df["month_num"] = df["date"].dt.month
df["cost_per_kg"] = (df["transportation_cost"] / df["shipment_volume_kg"]).round(3)

df.to_csv("logistics_dataset.csv", index=False)
print("Dataset simulated:", df.shape)
print(df.head())

# ----------------------------------------------------------------------
# 2. EXPLORATORY DATA ANALYSIS
# ----------------------------------------------------------------------
summary_stats = df[[
    "delivery_time_hrs", "promised_time_hrs", "delay_hrs",
    "shipment_volume_kg", "distance_km", "transportation_cost",
    "cost_per_kg", "customer_rating"
]].describe().T
summary_stats["skew"] = df[summary_stats.index].skew()
summary_stats.to_csv("summary_statistics.csv")
print("\n=== Summary Statistics ===")
print(summary_stats.round(2))

# On-time performance
otp_overall = 100 * (1 - df["delay_flag"].mean())
print(f"\nOverall On-Time Performance (OTP): {otp_overall:.1f}%")

otp_by_mode = 100 * (1 - df.groupby("transport_mode")["delay_flag"].mean())
otp_by_region = 100 * (1 - df.groupby("region")["delay_flag"].mean())
print("\nOTP by mode:\n", otp_by_mode.round(1))
print("\nOTP by region:\n", otp_by_region.round(1))

# Correlation matrix
corr_cols = ["distance_km", "shipment_volume_kg", "delivery_time_hrs",
             "delay_hrs", "transportation_cost", "cost_per_kg", "customer_rating"]
corr = df[corr_cols].corr()
corr.to_csv("correlation_matrix.csv")
print("\n=== Correlation Matrix ===")
print(corr.round(2))

# Average cost & delay by mode / region (pivot tables)
mode_summary = df.groupby("transport_mode").agg(
    avg_cost=("transportation_cost", "mean"),
    avg_cost_per_kg=("cost_per_kg", "mean"),
    avg_delivery_hrs=("delivery_time_hrs", "mean"),
    otp_pct=("delay_flag", lambda x: 100 * (1 - x.mean())),
    avg_rating=("customer_rating", "mean"),
    shipments=("shipment_id", "count"),
).round(2)
mode_summary.to_csv("mode_summary.csv")
print("\n=== Mode-level Summary ===")
print(mode_summary)

region_summary = df.groupby("region").agg(
    avg_cost=("transportation_cost", "mean"),
    avg_delivery_hrs=("delivery_time_hrs", "mean"),
    otp_pct=("delay_flag", lambda x: 100 * (1 - x.mean())),
    avg_rating=("customer_rating", "mean"),
    shipments=("shipment_id", "count"),
).round(2)
region_summary.to_csv("region_summary.csv")
print("\n=== Region-level Summary ===")
print(region_summary)

# ----------------------------------------------------------------------
# 3. VISUALIZATIONS
# ----------------------------------------------------------------------

# --- 3.1 Distribution of delivery times (histogram + KDE) ---------------
plt.figure(figsize=(8, 5))
sns.histplot(df["delivery_time_hrs"], bins=40, kde=True, color="#2C6E91")
plt.axvline(df["delivery_time_hrs"].mean(), color="red", linestyle="--", label=f"Mean = {df['delivery_time_hrs'].mean():.1f} hrs")
plt.axvline(df["delivery_time_hrs"].median(), color="orange", linestyle="--", label=f"Median = {df['delivery_time_hrs'].median():.1f} hrs")
plt.title("Distribution of Shipment Delivery Times")
plt.xlabel("Delivery Time (hours)")
plt.ylabel("Number of Shipments")
plt.legend()
plt.tight_layout()
plt.savefig(OUT + "01_delivery_time_distribution.png")
plt.close()

# --- 3.2 On-time performance by transport mode (bar chart) --------------
plt.figure(figsize=(8, 5))
ax = otp_by_mode.sort_values().plot(kind="barh", color=sns.color_palette("crest", 4))
plt.title("On-Time Performance (OTP) by Transport Mode")
plt.xlabel("On-Time Delivery Rate (%)")
plt.ylabel("Transport Mode")
for i, v in enumerate(otp_by_mode.sort_values()):
    plt.text(v + 0.5, i, f"{v:.1f}%", va="center")
plt.xlim(0, 100)
plt.tight_layout()
plt.savefig(OUT + "02_otp_by_mode.png")
plt.close()

# --- 3.3 Transportation cost vs. distance (scatter, colored by mode) ----
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df.sample(1200, random_state=1), x="distance_km", y="transportation_cost",
                 hue="transport_mode", alpha=0.6, s=35, palette="Set2")
plt.title("Transportation Cost vs. Distance by Mode")
plt.xlabel("Distance (km)")
plt.ylabel("Transportation Cost (USD)")
plt.legend(title="Mode", bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()
plt.savefig(OUT + "03_cost_vs_distance.png")
plt.close()

# --- 3.4 Correlation heatmap --------------------------------------------
plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, square=True,
            cbar_kws={"label": "Correlation coefficient"})
plt.title("Correlation Matrix of Key Logistics Metrics")
plt.tight_layout()
plt.savefig(OUT + "04_correlation_heatmap.png")
plt.close()

# --- 3.5 Monthly shipment volume & cost trend (dual-axis line chart) ----
monthly = df.groupby(["month_num", "month"]).agg(
    total_volume=("shipment_volume_kg", "sum"),
    total_cost=("transportation_cost", "sum"),
    shipments=("shipment_id", "count"),
).reset_index().sort_values("month_num")

fig, ax1 = plt.subplots(figsize=(9, 5))
ax2 = ax1.twinx()
ax1.plot(monthly["month"], monthly["shipments"], color="#2C6E91", marker="o", label="Shipment Count")
ax2.plot(monthly["month"], monthly["total_cost"], color="#D9622B", marker="s", label="Total Cost (USD)")
ax1.set_xlabel("Month (2025)")
ax1.set_ylabel("Shipment Count", color="#2C6E91")
ax2.set_ylabel("Total Transportation Cost (USD)", color="#D9622B")
ax1.tick_params(axis="y", labelcolor="#2C6E91")
ax2.tick_params(axis="y", labelcolor="#D9622B")
plt.title("Monthly Shipment Volume and Total Transportation Cost (2025)")
fig.tight_layout()
plt.savefig(OUT + "05_monthly_trend.png")
plt.close()

# --- 3.6 Cost breakdown by mode (stacked bar: fuel vs labor) ------------
cost_breakdown = df.groupby("transport_mode")[["fuel_cost", "labor_cost"]].mean()
plt.figure(figsize=(8, 5))
cost_breakdown.plot(kind="bar", stacked=True, color=["#D9622B", "#2C6E91"], figsize=(8, 5))
plt.title("Average Cost Composition by Transport Mode")
plt.ylabel("Average Cost per Shipment (USD)")
plt.xlabel("Transport Mode")
plt.xticks(rotation=0)
plt.legend(title="Cost Component")
plt.tight_layout()
plt.savefig(OUT + "06_cost_breakdown.png")
plt.close()

# --- 3.7 Delay distribution by region (boxplot) --------------------------
plt.figure(figsize=(8, 5))
order = df.groupby("region")["delay_hrs"].median().sort_values(ascending=False).index
sns.boxplot(data=df, x="region", y="delay_hrs", order=order, palette="flare")
plt.axhline(0, color="black", linestyle="--", linewidth=1)
plt.title("Delivery Delay Distribution by Region\n(Delay hrs = Actual − Promised)")
plt.xlabel("Region")
plt.ylabel("Delay (hours)")
plt.tight_layout()
plt.savefig(OUT + "07_delay_by_region.png")
plt.close()

# --- 3.8 Customer rating vs delay (scatter + trend) ----------------------
plt.figure(figsize=(8, 5))
sns.regplot(data=df.sample(1500, random_state=2), x="delay_hrs", y="customer_rating",
            scatter_kws={"alpha": 0.3, "s": 20}, line_kws={"color": "red"})
plt.title("Customer Satisfaction Rating vs. Delivery Delay")
plt.xlabel("Delay (hours, negative = early)")
plt.ylabel("Customer Rating (1-5)")
plt.tight_layout()
plt.savefig(OUT + "08_rating_vs_delay.png")
plt.close()

print("\nAll visualizations saved to the images/ directory.")
