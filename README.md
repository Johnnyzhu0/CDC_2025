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
- **Output**: Monte Carlo simulations with percentile bands (50%, 90%)
- **What it models**: Random walk behavior with drift and volatility
- **Use case**: Risk assessment and probabilistic forecasting

### 3. **Profit & Productivity Analysis**
**Purpose**: Calculate economic efficiency metrics
- **Input**: Value Added, Employment, Compensation data
- **Output**: Labor productivity and profit margin calculations
- **What it models**: Economic efficiency and profitability trends
- **Use case**: Assessing sector performance and competitiveness

## 📈 Visualizations Guide

### 1. **Economic Output Indicators**
Shows historical trends of RealGrossOutput and RealValueAdded from 2012-2023
- **Circle markers**: Economic output metrics in millions
- **Separate scaling**: Optimized for large-value economic indicators

### 2. **Labor Market Indicators**
Shows historical trends of Employment and Compensation from 2012-2023
- **Square markers**: Labor market metrics
- **Separate scaling**: Optimized for employment and compensation values

### 3. **Economic Indicators Growth Rates**
Displays year-over-year percentage changes for all indicators EXCEPT Compensation
- **Circle markers**: Economic growth rates with stable scaling
- **Positive values**: Growth periods
- **Negative values**: Decline periods
- **Red dashed line**: Zero growth reference
- **Clean visualization**: Without compensation's high volatility

### 4. **Compensation Growth Rate**
Dedicated chart for compensation growth rate (separated due to high volatility)
- **Square markers**: Orange color for distinction
- **Separate scaling**: Handles extreme swings (-50% to +200%)
- **Blue dotted line**: Mean growth rate reference
- **High volatility**: Shows 70.29% standard deviation in growth

### 5. **ARIMA Forecasts (2024-2030)**
Shows model predictions with uncertainty bands
- **Solid line**: Historical data
- **Dashed line**: ARIMA forecasts
- **Shaded area**: Confidence intervals
- **Annotations**: Key forecast years and values

### 6. **GBM Simulation Paths**
Displays probabilistic forecasts from Monte Carlo simulations
- **Solid line**: Historical data
- **Central line**: Median forecast (50th percentile)
- **Light shading**: 90% confidence band (5th-95th percentiles)
- **Dark shading**: 50% confidence band (25th-75th percentiles)

### 7. **Labor Productivity Analysis**
Tracks economic efficiency over time
- **Green line with circles**: Labor productivity (output per worker)
- **Dedicated scaling**: Optimized for productivity values
- **Trend analysis**: Shows 4.63% annual growth

### 8. **Labor Cost Ratio Analysis**
Shows labor compensation as percentage of total output
- **Red line with squares**: Labor cost ratio (formerly "profit margin")
- **Corrected calculation**: Compensation / Output ratio
- **Economic interpretation**: 0.2% average ratio (highly capital-intensive sector)
- **Black dashed line**: Zero reference

### 9. **ARIMA Model Residuals (Outlier Removed)**
Shows model fit quality and assumptions with 2012 outlier excluded
- **Points**: Residual values (actual - predicted) from 2013-2023
- **Red dashed line**: Zero reference
- **Pattern**: Random scatter indicates good model fit
- **Improvement**: 2012 outlier removed for cleaner analysis

### 10. **Correlation Matrix Heatmap**
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
- **Growth trends** in the U.S. space economy with separated visualization for stable vs. volatile indicators
- **Forecasted values** for 2024-2030 using multiple methodologies (ARIMA & GBM)
- **Risk assessments** through probabilistic modeling
- **Labor productivity trends** showing 4.63% annual growth
- **Labor cost efficiency** revealing 0.2% labor cost ratio (highly capital-intensive sector)
- **Compensation volatility** with 70.29% standard deviation requiring separate analysis
- **Improved model diagnostics** with outlier handling and dedicated residual analysis

---
*Carolina Data Challenge 2025* 🗣️

