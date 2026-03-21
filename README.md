# Kenya Inflation Risk Audit (1960–2024)

## A Statistical and Policy Analysis Report

This project is a simulation of a central bank data science workflow, combining quantitative analysis with policy-oriented interpretation.

The objective is to audit Kenya’s inflation history (1960–2024) using statistical methods and translate the findings into actionable policy insights.

---

## Project Overview

This project approaches inflation as a **risk distribution problem** rather than a routine macroeconomic indicator.

Using a Z-score framework, the analysis identifies abnormal inflation periods, evaluates volatility, and assesses whether Kenya’s inflation follows a normal or shock-prone distribution.

The output is structured in two parts:
- A **policy memorandum** summarizing key findings and implications
- A **technical notebook** providing the statistical foundation for the analysis

---

## Key Findings

- The long-run average inflation rate is approximately **9.7%**
- Inflation exhibits **high volatility** (σ ≈ 7.98%)
- The distribution is **positively skewed**, indicating a higher likelihood of extreme upward shocks
- Major outlier events identified:
  - 1993 Hyperinflation (Z ≈ 4.54)
  - 2008 Crisis (Z ≈ 2.07)

### Interpretation

Kenya’s inflation does not follow a normal distribution.  
It is better understood as a **shock-prone system with fat-tailed risk characteristics**.

---

## Methodology

The analysis is based on core statistical concepts:

- Mean (μ) to estimate the long-term inflation anchor  
- Standard Deviation (σ) to measure volatility  
- Z-scores to identify extreme deviations from historical norms  

### Risk Thresholds

- Z > 2 → Extreme inflationary pressure  
- Z < -2 → Deflationary anomaly (rare in Kenya)

---

## Technical Stack

- Python  
- pandas  
- numpy  
- matplotlib  
- seaborn  
- scipy  

---

## Repository Structure
