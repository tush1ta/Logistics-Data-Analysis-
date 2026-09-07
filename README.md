# Advanced Data Analysis and Visualization in Logistics

**Week 3 Task — Exploratory Data Analysis, Visualization, and Operational Insights for a Logistics Network**

## Overview

This project simulates a realistic, shipment-level logistics operations dataset and performs an end-to-end analytical workflow in Python: data generation, exploratory data analysis (EDA), visualization, and interpretation. The goal is to demonstrate how structured analysis of delivery-time, cost, and volume metrics can uncover operational bottlenecks, cost drivers, and customer-experience risks in a multi-modal freight network.

Since a proprietary logistics dataset wasn't available, a synthetic dataset was built with parameters (speed profiles, cost structures, regional congestion factors) modeled on typical multi-modal freight operations — so the same pipeline applies directly to a real operational dataset with the same schema.

## Dataset

`logistics_dataset.csv` — 5,000 simulated shipment records for FY2025, across:
- **5 regional hubs:** North, South, East, West, Central
- **4 transport modes:** Road, Rail, Air, Sea

**Key fields:** `shipment_id`, `date`, `region`, `transport_mode`, `shipment_volume_kg`, `distance_km`, `delivery_time_hrs`, `promised_time_hrs`, `delay_hrs`, `delay_flag`, `transportation_cost`, `fuel_cost`, `labor_cost`, `customer_rating`, `cost_per_kg`.

## What's in this repo

| File | Description |
|---|---|
| `logistics_analysis.py` | Full Python script: dataset simulation, EDA (summary stats, correlations, group-bys), and generation of all 8 visualizations |
| `logistics_dataset.csv` | The generated dataset used for the analysis |
| `Logistics_Data_Analysis_Report.docx` | Full written report — methodology, code excerpts, embedded charts, and analytical recommendations |
| `images/` | All 8 charts saved as standalone PNG files |

## Methodology

Built with **pandas**, **numpy**, **matplotlib**, and **seaborn**:
1. **Simulate** shipment-level data with mode-specific speed/cost profiles and regional congestion factors
2. **Explore** with `.describe()`, skewness, on-time-performance (OTP) rates, and correlation matrices
3. **Visualize**: delivery-time distribution, OTP by mode, cost vs. distance, correlation heatmap, monthly trend, cost composition, delay by region, and rating vs. delay
4. **Interpret** each chart in terms of operational efficiency, cost drivers, and bottlenecks

## Key Findings

- Overall on-time performance (OTP) is **69.4%**
- **Air** is fastest per shipment but least reliable (63.7% OTP) and ~15x more expensive per kg than Road
- **Central** and **East** regions show the highest delivery delay — candidates for operational review
- **Delay is the strongest driver of customer satisfaction** (r = -0.43), more than cost, distance, or volume

## How to Run

```bash
pip install pandas numpy matplotlib seaborn
python logistics_analysis.py
```

This regenerates `logistics_dataset.csv`, prints EDA summaries to the console, and writes all charts to `images/`.

## Tools Used

Python 3, pandas, numpy, matplotlib, seaborn

## License

This project uses synthetic data and is intended for educational/portfolio purposes.