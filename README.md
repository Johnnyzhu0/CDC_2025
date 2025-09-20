# U.S. Space Economy Analysis - Carolina Data Challenge 2025

This project analyzes U.S. Space Economy data (2012-2023) using advanced statistical models to understand trends, relationships, and forecast future growth through 2030.

## 📊 Data Overview

The analysis uses data from Business.xlsx containing 8 tables of space economy indicators:
- **Value Added** (Real & Nominal)
- **Gross Output** (Real & Nominal) 
- **Price Indexes**
- **Employment**
- **Compensation**

## 🔬 Statistical Models & Analysis

### 1. **ARIMA Models (AutoRegressive Integrated Moving Average)**
**Purpose**: Time series forecasting with trend and seasonality analysis
- **Input**: Historical time series data (2012-2023)
- **Output**: Forecasts for 2024-2030 with confidence intervals
- **What it models**: Linear trends, autocorrelations, and moving averages in the data
- **Use case**: Predicting future values based on historical patterns

### 2. **Geometric Brownian Motion (GBM)**
**Purpose**: Stochastic modeling for financial/economic forecasting
- **Input**: Historical returns and volatility
- **Output**: Monte Carlo simulations with percentile bands (5th, 25th, 50th, 75th, 95th)
- **What it models**: Random walk behavior with drift and volatility
- **Use case**: Risk assessment and probabilistic forecasting

### 3. **Granger Causality Tests**
**Purpose**: Identify causal relationships between economic indicators
- **Input**: Multiple time series (differenced to ensure stationarity)
- **Output**: Statistical significance of X causing Y relationships
- **What it models**: Whether past values of one variable help predict another
- **Use case**: Understanding economic interdependencies

### 4. **Profit & Productivity Analysis**
**Purpose**: Calculate economic efficiency metrics
- **Input**: Value Added, Employment, Compensation data
- **Output**: Labor productivity and profit margin calculations
- **What it models**: Economic efficiency and profitability trends
- **Use case**: Assessing sector performance and competitiveness

## 📈 Visualizations Guide

### 1. **Time Series Overview**
Shows historical trends of key indicators from 2012-2023
- **Red lines**: Major economic indicators
- **Markers**: Data points for each year

### 2. **Annual Growth Rates**
Displays year-over-year percentage changes
- **Positive values**: Growth periods
- **Negative values**: Decline periods
- **Red dashed line**: Zero growth reference

### 3. **ARIMA Forecasts (2024-2030)**
Shows model predictions with uncertainty bands
- **Solid line**: Historical data
- **Dashed line**: ARIMA forecasts
- **Shaded area**: Confidence intervals
- **Annotations**: Key forecast years and values

### 4. **GBM Simulation Paths**
Displays probabilistic forecasts from Monte Carlo simulations
- **Solid line**: Historical data
- **Central line**: Median forecast (50th percentile)
- **Light shading**: 90% confidence band (5th-95th percentiles)
- **Dark shading**: 50% confidence band (25th-75th percentiles)

### 5. **Granger Causality Results**
Bar chart showing significant causal relationships
- **Bar length**: Strength of statistical significance (-log10 p-value)
- **Red line**: 5% significance threshold
- **Orange line**: 10% significance threshold
- **Labels**: p-values for each relationship

### 6. **Productivity & Profitability Metrics**
Tracks economic efficiency over time
- **Green line**: Labor productivity (output per worker)
- **Red line**: Profit margins (percentage)

### 7. **ARIMA Model Residuals**
Shows model fit quality and assumptions
- **Points**: Residual values (actual - predicted)
- **Red dashed line**: Zero reference
- **Pattern**: Random scatter indicates good model fit

### 8. **Correlation Matrix Heatmap**
**🔴 Red colors**: **Positive correlations**
- Darker red = stronger positive correlation (closer to +1.0)
- When one variable increases, the other tends to increase
- Example: Value Added and Gross Output typically move together

**🔵 Blue colors**: **Negative correlations**
- Darker blue = stronger negative correlation (closer to -1.0) 
- When one variable increases, the other tends to decrease
- Less common in economic data

**⚪ White/Light colors**: **Weak or no correlation** (close to 0)
- Variables with no clear linear relationship

The `'coolwarm'` colormap is centered at zero correlation, with values ranging from -1.0 to +1.0. Numbers in each cell show exact correlation coefficients.

## 🚀 How to Run

```bash
# Activate virtual environment
source .venv/bin/activate

# Install dependencies
pip install pandas openpyxl matplotlib seaborn statsmodels scipy

# Run analysis
python main.py
```

## 📁 Project Structure

```
CDC_2025/
├── main.py                    # Main analysis script
├── Business.xlsx              # Raw data file
├── table_X_clean.txt         # Cleaned data exports
├── data_analysis_results.txt  # Analysis output
├── space_economy_comprehensive_analysis.png  # Visualizations
└── README.md                 # This documentation
```

## 📊 Key Findings

The analysis provides insights into:
- **Growth trends** in the U.S. space economy
- **Forecasted values** for 2024-2030 using multiple methodologies
- **Causal relationships** between economic indicators
- **Risk assessments** through probabilistic modeling
- **Productivity trends** and economic efficiency

---
*Carolina Data Challenge 2025* 🗣️

