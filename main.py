# Space Economy Data Analysis - Carolina Data Challenge 2025
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('default')
sns.set_palette("husl")

class SpaceEconomyAnalyzer:
    """Comprehensive analysis of U.S. Space Economy data"""
    
    def __init__(self):
        self.data = {}
        self.time_series = pd.DataFrame()
        self.results = {}
        
    def load_data_from_txt_files(self):
        """Load and parse data from clean .txt files"""
        print("Loading data from clean text files...")
        
        table_mapping = {
            'table_1_clean.txt': 'RealValueAdded',
            'table_2_clean.txt': 'NominalValueAdded', 
            'table_3_clean.txt': 'ValueAddedPriceIndex',
            'table_4_clean.txt': 'RealGrossOutput',
            'table_5_clean.txt': 'NominalGrossOutput',
            'table_6_clean.txt': 'GrossOutputPriceIndex',
            'table_7_clean.txt': 'Employment',
            'table_8_clean.txt': 'Compensation'
        }
        
        for file_name, series_name in table_mapping.items():
            try:
                # Read the clean text file
                with open(file_name, 'r') as f:
                    lines = f.readlines()
                
                # Find the start of the full data section
                start_idx = None
                for i, line in enumerate(lines):
                    if 'FULL DATA:' in line:
                        start_idx = i + 2  # Skip the separator line
                        break
                
                if start_idx is None:
                    print(f"Warning: Could not find data section in {file_name}")
                    continue
                
                # Parse the tabular data
                data_lines = [line.strip() for line in lines[start_idx:] if line.strip()]
                
                # Extract header (years)
                header_line = data_lines[0]
                columns = header_line.split()
                years = [col.strip() for col in columns if col.strip().isdigit() and len(col.strip()) == 4]
                
                if not years:
                    print(f"Warning: Could not find year columns in {file_name}")
                    continue
                
                # Look for space economy total row or aggregate data
                total_row = None
                
                # Method 1: Look for a row that mentions "space economy" or similar
                for line in data_lines[1:]:
                    if any(keyword in line.lower() for keyword in ['space economy', 'total', 'all industries']):
                        parts = line.split()
                        # Try to extract numeric values from the end
                        numeric_data = []
                        for part in parts[-len(years):]:
                            try:
                                if part != '…' and part != 'nan' and part.replace(',', '').replace('.', '').isdigit():
                                    numeric_data.append(float(part.replace(',', '')))
                                else:
                                    numeric_data.append(np.nan)
                            except:
                                numeric_data.append(np.nan)
                        
                        if len(numeric_data) == len(years) and not all(pd.isna(numeric_data)):
                            total_row = numeric_data
                            break
                
                # Method 2: If no explicit total found, look for the first row with substantial numeric data
                if total_row is None:
                    for line in data_lines[1:]:
                        parts = line.split()
                        if len(parts) >= len(years) + 2:  # Index + description + data
                            # Extract numeric values from the end
                            numeric_data = []
                            for part in parts[-len(years):]:
                                try:
                                    if part != '…' and part != 'nan' and part.replace(',', '').replace('.', '').isdigit():
                                        value = float(part.replace(',', ''))
                                        # Only accept values that seem reasonable (not too small)
                                        if value > 10:  # Reasonable threshold for space economy data
                                            numeric_data.append(value)
                                        else:
                                            numeric_data.append(np.nan)
                                    else:
                                        numeric_data.append(np.nan)
                                except:
                                    numeric_data.append(np.nan)
                            
                            # Check if we have enough valid data points
                            valid_points = sum(1 for x in numeric_data if not pd.isna(x))
                            if len(numeric_data) == len(years) and valid_points >= len(years) * 0.5:
                                total_row = numeric_data
                                break
                
                # Method 3: Try to find manufacturing row as a proxy for major space economy component
                if total_row is None and series_name in ['Employment', 'Compensation']:
                    for line in data_lines[1:]:
                        if 'Manufacturing' in line or 'Computer and electronic products' in line:
                            parts = line.split()
                            if len(parts) >= len(years) + 2:
                                numeric_data = []
                                for part in parts[-len(years):]:
                                    try:
                                        if part != '…' and part != 'nan' and part.replace(',', '').replace('.', '').isdigit():
                                            value = float(part.replace(',', ''))
                                            numeric_data.append(value)
                                        else:
                                            numeric_data.append(np.nan)
                                    except:
                                        numeric_data.append(np.nan)
                                
                                valid_points = sum(1 for x in numeric_data if not pd.isna(x))
                                if len(numeric_data) == len(years) and valid_points >= len(years) * 0.5:
                                    total_row = numeric_data
                                    print(f"  Using Manufacturing data as proxy for {series_name}")
                                    break
                
                if total_row is not None:
                    # Create time series
                    year_indices = pd.to_datetime([int(y) for y in years], format='%Y')
                    series = pd.Series(total_row, index=year_indices)
                    series = series.replace(0, np.nan)  # Replace zeros with NaN for log calculations
                    self.data[series_name] = series
                    print(f"✅ Loaded {series_name}: {len(series.dropna())} data points")
                else:
                    print(f"❌ Could not extract data from {file_name}")
                    
            except Exception as e:
                print(f"❌ Error loading {file_name}: {e}")
        
        # Combine into DataFrame
        self.time_series = pd.DataFrame(self.data)
        self.time_series = self.time_series.dropna(how='all')
        
        print(f"\n📊 Combined dataset shape: {self.time_series.shape}")
        print(f"Available series: {list(self.time_series.columns)}")
        
        return self.time_series
    
    def create_transformations(self):
        """Create log and difference transformations"""
        print("\nCreating log and difference transformations...")
        
        df_trans = pd.DataFrame(index=self.time_series.index)
        
        for col in self.time_series.columns:
            # Log transformation (only for positive values)
            series = self.time_series[col].replace(0, np.nan)
            valid_data = series[series > 0]
            
            if len(valid_data) > 0:
                df_trans[f'{col}_log'] = np.log(series)
                df_trans[f'{col}_dlog'] = df_trans[f'{col}_log'].diff()
                df_trans[f'{col}_pct_change'] = series.pct_change()
        
        self.transformations = df_trans
        return df_trans
    
    def stationarity_tests(self):
        """Perform Augmented Dickey-Fuller tests"""
        print("\nPerforming stationarity tests (ADF)...")
        
        adf_results = []
        
        for col in self.time_series.columns:
            # Test levels
            series = self.time_series[col].dropna()
            if len(series) >= 4:
                try:
                    result = adfuller(series, autolag='AIC')
                    adf_results.append({
                        'Series': col,
                        'Transform': 'levels',
                        'ADF_stat': result[0],
                        'p_value': result[1],
                        'Stationary': result[1] < 0.05
                    })
                except:
                    adf_results.append({
                        'Series': col, 'Transform': 'levels',
                        'ADF_stat': np.nan, 'p_value': np.nan, 'Stationary': False
                    })
            
            # Test first differences
            if f'{col}_dlog' in self.transformations.columns:
                diff_series = self.transformations[f'{col}_dlog'].dropna()
                if len(diff_series) >= 3:
                    try:
                        result = adfuller(diff_series, autolag='AIC')
                        adf_results.append({
                            'Series': col,
                            'Transform': 'first_diff',
                            'ADF_stat': result[0],
                            'p_value': result[1],
                            'Stationary': result[1] < 0.05
                        })
                    except:
                        adf_results.append({
                            'Series': col, 'Transform': 'first_diff',
                            'ADF_stat': np.nan, 'p_value': np.nan, 'Stationary': False
                        })
        
        self.adf_results = pd.DataFrame(adf_results)
        return self.adf_results
    
    def fit_arima_models(self, forecast_periods=7):
        """Fit ARIMA models for forecasting 2024-2030"""
        print(f"\nFitting ARIMA models (forecasting 2024-2030)...")
        
        arima_results = {}
        
        key_series = ['RealGrossOutput', 'Employment', 'Compensation', 'RealValueAdded']
        
        for series_name in key_series:
            if series_name in self.time_series.columns:
                series = self.time_series[series_name].dropna()
                
                if len(series) >= 8:
                    print(f"  Fitting ARIMA for {series_name}...")
                    
                    # Determine integration order based on stationarity
                    d = 1  # Default to first difference
                    if hasattr(self, 'adf_results'):
                        level_test = self.adf_results[
                            (self.adf_results['Series'] == series_name) & 
                            (self.adf_results['Transform'] == 'levels')
                        ]
                        if not level_test.empty and level_test.iloc[0]['Stationary']:
                            d = 0
                    
                    # Grid search for best ARIMA model
                    best_aic = np.inf
                    best_model = None
                    best_order = None
                    
                    for p in range(0, min(3, len(series)//3)):
                        for q in range(0, min(3, len(series)//3)):
                            try:
                                model = ARIMA(series, order=(p, d, q))
                                fitted_model = model.fit()
                                
                                if fitted_model.aic < best_aic:
                                    best_aic = fitted_model.aic
                                    best_model = fitted_model
                                    best_order = (p, d, q)
                            except:
                                continue
                    
                    if best_model is not None:
                        # Generate forecasts for 2024-2030 (7 years from 2023)
                        try:
                            forecast = best_model.get_forecast(steps=forecast_periods)
                            
                            # Create explicit forecast years 2024-2030
                            forecast_years = [2024, 2025, 2026, 2027, 2028, 2029, 2030]
                            forecast_index = pd.to_datetime(forecast_years, format='%Y')
                            
                            # Get forecast values and confidence intervals
                            forecast_mean = pd.Series(forecast.predicted_mean.values, index=forecast_index)
                            forecast_ci = pd.DataFrame(forecast.conf_int().values, 
                                                     index=forecast_index,
                                                     columns=['lower_ci', 'upper_ci'])
                            
                            arima_results[series_name] = {
                                'model': best_model,
                                'order': best_order,
                                'aic': best_aic,
                                'forecast_mean': forecast_mean,
                                'forecast_ci': forecast_ci,
                                'forecast_years': forecast_years,
                                'fitted_values': best_model.fittedvalues,
                                'residuals': best_model.resid,
                                'last_observed_year': series.index[-1].year,
                                'last_observed_value': series.iloc[-1]
                            }
                            
                            # Print forecast summary
                            print(f"    ✅ {series_name} ARIMA{best_order} forecasts:")
                            for year, value in zip(forecast_years, forecast_mean.values):
                                print(f"       {year}: {value:,.0f}")
                                
                        except Exception as e:
                            print(f"    ❌ Forecast error for {series_name}: {e}")
                    else:
                        print(f"    ❌ Could not fit ARIMA for {series_name}")
        
        self.arima_results = arima_results
        return arima_results
    
    def geometric_brownian_motion(self, n_simulations=1000, forecast_periods=7):
        """Fit Geometric Brownian Motion model for 2024-2030"""
        print(f"\nFitting Geometric Brownian Motion (GBM) models for 2024-2030...")
        
        gbm_results = {}
        
        # Focus on monetary series for GBM
        monetary_series = ['NominalGrossOutput', 'NominalValueAdded', 'Compensation']
        
        for series_name in monetary_series:
            if series_name in self.time_series.columns:
                series = self.time_series[series_name].dropna()
                
                if len(series) >= 5:
                    print(f"  Fitting GBM for {series_name}...")
                    
                    # Calculate log returns
                    log_returns = np.log(series / series.shift(1)).dropna()
                    
                    # Estimate parameters
                    mu = log_returns.mean()  # Drift
                    sigma = log_returns.std()  # Volatility
                    
                    # Monte Carlo simulation
                    np.random.seed(42)  # For reproducibility
                    dt = 1.0  # Annual time step
                    S0 = series.iloc[-1]  # Last observed value (2023)
                    
                    # Generate forecast paths for 2024-2030 (7 years)
                    forecast_paths = np.zeros((n_simulations, forecast_periods))
                    
                    for i in range(n_simulations):
                        path = [S0]
                        for t in range(forecast_periods):
                            dW = np.random.normal(0, np.sqrt(dt))
                            S_next = path[-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * dW)
                            path.append(S_next)
                        forecast_paths[i, :] = path[1:]
                    
                    # Calculate percentiles
                    percentiles = [5, 25, 50, 75, 95]
                    forecast_percentiles = np.percentile(forecast_paths, percentiles, axis=0)
                    
                    # Create explicit forecast years 2024-2030
                    forecast_years = [2024, 2025, 2026, 2027, 2028, 2029, 2030]
                    forecast_index = pd.to_datetime(forecast_years, format='%Y')
                    
                    gbm_results[series_name] = {
                        'mu': mu,
                        'sigma': sigma,
                        'log_returns': log_returns,
                        'forecast_paths': forecast_paths,
                        'forecast_percentiles': pd.DataFrame(
                            forecast_percentiles.T,
                            index=forecast_index,
                            columns=[f'p{p}' for p in percentiles]
                        ),
                        'forecast_years': forecast_years,
                        'last_value': S0,
                        'last_observed_year': series.index[-1].year
                    }
                    
                    # Print GBM forecast summary
                    print(f"    ✅ {series_name} GBM forecasts (median):")
                    median_forecasts = forecast_percentiles[2]  # 50th percentile
                    for year, value in zip(forecast_years, median_forecasts):
                        print(f"       {year}: {value:,.0f}")
        
        self.gbm_results = gbm_results
        return gbm_results
    
    def profit_analysis(self):
        """Analyze profit-related metrics"""
        print("\nPerforming profit analysis...")
        
        profit_results = {}
        
        # Calculate profit proxies and productivity metrics
        if 'RealGrossOutput' in self.time_series.columns and 'Employment' in self.time_series.columns:
            # Labor productivity
            labor_productivity = self.time_series['RealGrossOutput'] / self.time_series['Employment']
            labor_productivity.name = 'LaborProductivity'
            
            # Growth rates
            labor_prod_growth = labor_productivity.pct_change()
            
            profit_results['labor_productivity'] = {
                'series': labor_productivity,
                'growth_rate': labor_prod_growth,
                'avg_growth': labor_prod_growth.mean(),
                'volatility': labor_prod_growth.std()
            }
        
        if 'Compensation' in self.time_series.columns and 'Employment' in self.time_series.columns:
            # Average compensation per employee
            avg_compensation = self.time_series['Compensation'] / self.time_series['Employment']
            avg_compensation.name = 'AvgCompensation'
            
            comp_growth = avg_compensation.pct_change()
            
            profit_results['avg_compensation'] = {
                'series': avg_compensation,
                'growth_rate': comp_growth,
                'avg_growth': comp_growth.mean(),
                'volatility': comp_growth.std()
            }
        
        # Labor cost ratio (if we have both output and compensation)
        if 'RealGrossOutput' in self.time_series.columns and 'Compensation' in self.time_series.columns:
            # Calculate labor cost as percentage of total output
            # This shows what percentage of revenue goes to labor compensation
            real_output = self.time_series['RealGrossOutput']
            compensation = self.time_series['Compensation']
            
            # Calculate labor cost ratio (compensation / output)
            labor_cost_ratio = compensation / real_output
            labor_cost_ratio.name = 'LaborCostRatio'
            
            ratio_change = labor_cost_ratio.diff()
            
            profit_results['profit_margin'] = {
                'series': labor_cost_ratio,  # Keep same key for compatibility
                'change': ratio_change,
                'avg_margin': labor_cost_ratio.mean(),
                'trend': stats.linregress(range(len(labor_cost_ratio.dropna())), labor_cost_ratio.dropna()).slope
            }
        
        self.profit_results = profit_results
        return profit_results
    
    def create_visualizations(self):
        """Create comprehensive visualizations"""
        print("\nCreating visualizations...")
        
        # Check if we have any data to plot
        if self.time_series.empty:
            print("❌ No data available for visualization")
            return
        
        # Filter out empty series
        available_series = {}
        for col in ['RealGrossOutput', 'Employment', 'Compensation', 'RealValueAdded']:
            if col in self.time_series.columns:
                series_data = self.time_series[col].dropna()
                if len(series_data) > 0:
                    available_series[col] = series_data
        
        if not available_series:
            print("❌ No valid data series available for visualization")
            return
        
        print(f"📊 Creating plots for: {list(available_series.keys())}")
        
        # Determine layout based on available plots
        n_plots = 0
        if available_series: n_plots += 2  # Time series overview (now split into 2 plots)
        if hasattr(self, 'transformations') and available_series: n_plots += 2  # Growth rates (now split into 2 plots)
        if hasattr(self, 'arima_results') and self.arima_results: n_plots += 1  # ARIMA
        if hasattr(self, 'gbm_results') and self.gbm_results: n_plots += 1  # GBM
        if hasattr(self, 'profit_results') and self.profit_results: n_plots += 2  # Profit (now split into 2 plots)
        if hasattr(self, 'arima_results') and self.arima_results: n_plots += 1  # Residuals
        if len(available_series) >= 2: n_plots += 1  # Correlation
        
        if n_plots == 0:
            print("❌ No plots to generate")
            return
        
        # Set up the plotting layout
        rows = max(2, (n_plots + 1) // 2)
        fig = plt.figure(figsize=(20, 6 * rows))
        plot_num = 1
        
        # 1. Economic Output Indicators (RealGrossOutput and RealValueAdded)
        if available_series:
            output_series = {}
            for col in ['RealGrossOutput', 'RealValueAdded']:
                if col in available_series:
                    output_series[col] = available_series[col]
            
            if output_series:
                plt.subplot(rows, 2, plot_num)
                for col, series_data in output_series.items():
                    plt.plot(series_data.index, series_data, marker='o', label=col, linewidth=2)
                plt.title('Economic Output Indicators (2012-2023)', fontsize=14, fontweight='bold')
                plt.xlabel('Year')
                plt.ylabel('Value (Millions)')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plot_num += 1
        
        # 2. Labor Market Indicators (Employment and Compensation)
        if available_series:
            labor_series = {}
            for col in ['Employment', 'Compensation']:
                if col in available_series:
                    labor_series[col] = available_series[col]
            
            if labor_series:
                plt.subplot(rows, 2, plot_num)
                for col, series_data in labor_series.items():
                    plt.plot(series_data.index, series_data, marker='s', label=col, linewidth=2)
                plt.title('Labor Market Indicators (2012-2023)', fontsize=14, fontweight='bold')
                plt.xlabel('Year')
                plt.ylabel('Value')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plot_num += 1
        
        # 3. Economic Indicators Growth Rates (excluding Compensation)
        if hasattr(self, 'transformations') and available_series:
            growth_data = {}
            for col in available_series.keys():
                if col != 'Compensation' and f'{col}_pct_change' in self.transformations.columns:
                    growth = self.transformations[f'{col}_pct_change'].dropna() * 100
                    if len(growth) > 0:
                        growth_data[col] = growth
            
            if growth_data:
                plt.subplot(rows, 2, plot_num)
                for col, growth in growth_data.items():
                    plt.plot(growth.index, growth, marker='o', label=f'{col} Growth %', linewidth=2)
                plt.title('Economic Indicators Growth Rates', fontsize=14, fontweight='bold')
                plt.xlabel('Year')
                plt.ylabel('Growth Rate (%)')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.axhline(y=0, color='red', linestyle='--', alpha=0.5)
                plot_num += 1
        
        # 4. Compensation Growth Rate (separate due to high volatility)
        if hasattr(self, 'transformations') and 'Compensation' in available_series:
            if 'Compensation_pct_change' in self.transformations.columns:
                comp_growth = self.transformations['Compensation_pct_change'].dropna() * 100
                if len(comp_growth) > 0:
                    plt.subplot(rows, 2, plot_num)
                    plt.plot(comp_growth.index, comp_growth, marker='s', label='Compensation Growth %', 
                            linewidth=2, color='orange', markersize=6)
                    plt.title('Compensation Growth Rate', fontsize=14, fontweight='bold')
                    plt.xlabel('Year')
                    plt.ylabel('Growth Rate (%)')
                    plt.legend()
                    plt.grid(True, alpha=0.3)
                    plt.axhline(y=0, color='red', linestyle='--', alpha=0.5)
                    # Add mean line for reference
                    mean_growth = comp_growth.mean()
                    plt.axhline(y=mean_growth, color='blue', linestyle=':', alpha=0.7, 
                              label=f'Mean: {mean_growth:.1f}%')
                    plt.legend()
                    plot_num += 1
        
        # 5. ARIMA Forecasts (only if we have ARIMA results)
        if hasattr(self, 'arima_results') and self.arima_results:
            # Find a series to display
            arima_series = None
            for series_name in ['RealGrossOutput', 'RealValueAdded', 'Compensation']:
                if series_name in self.arima_results and series_name in available_series:
                    arima_series = series_name
                    break
            
            if arima_series:
                plt.subplot(rows, 2, plot_num)
                results = self.arima_results[arima_series]
                historical = self.time_series[arima_series].dropna()
                forecast = results['forecast_mean']
                
                plt.plot(historical.index, historical, 'o-', label=f'{arima_series} (Historical)', linewidth=2)
                plt.plot(forecast.index, forecast, 's--', label=f'{arima_series} (ARIMA 2024-2030)', linewidth=2)
                
                # Add confidence interval
                ci = results['forecast_ci']
                plt.fill_between(forecast.index, ci['lower_ci'], ci['upper_ci'], alpha=0.2)
                
                # Add forecast year labels for key years
                key_years = [2024, 2026, 2028, 2030]
                for year in key_years:
                    if year in results['forecast_years']:
                        idx = results['forecast_years'].index(year)
                        value = forecast.iloc[idx]
                        plt.annotate(f'{year}\n{value:,.0f}', 
                                   xy=(pd.to_datetime(str(year)), value),
                                   xytext=(5, 5), textcoords='offset points',
                                   fontsize=8, ha='left')
                
                plt.title(f'ARIMA Forecasts 2024-2030 - {arima_series}', fontsize=14, fontweight='bold')
                plt.xlabel('Year')
                plt.ylabel('Value (Millions)')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plot_num += 1
        
        # 6. GBM Simulation Paths (only if we have GBM results)
        if hasattr(self, 'gbm_results') and self.gbm_results:
            # Find a series to display
            gbm_series = None
            for series_name in ['NominalGrossOutput', 'NominalValueAdded', 'Compensation']:
                if series_name in self.gbm_results and series_name in available_series:
                    gbm_series = series_name
                    break
            
            if gbm_series:
                plt.subplot(rows, 2, plot_num)
                results = self.gbm_results[gbm_series]
                historical = self.time_series[gbm_series].dropna()
                percentiles = results['forecast_percentiles']
                
                plt.plot(historical.index, historical, 'o-', label=f'{gbm_series} (Historical)', linewidth=2)
                
                # Plot percentile bands
                plt.plot(percentiles.index, percentiles['p50'], 's-', label='GBM Median (2024-2030)', linewidth=2)
                plt.fill_between(percentiles.index, percentiles['p5'], percentiles['p95'], 
                               alpha=0.2, label='90% Confidence Band')
                plt.fill_between(percentiles.index, percentiles['p25'], percentiles['p75'], 
                               alpha=0.3, label='50% Confidence Band')
                
                # Add forecast year labels for key years
                key_years = [2024, 2026, 2028, 2030]
                for year in key_years:
                    if year in results['forecast_years']:
                        idx = results['forecast_years'].index(year)
                        value = percentiles['p50'].iloc[idx]
                        plt.annotate(f'{year}\n{value:,.0f}', 
                                   xy=(pd.to_datetime(str(year)), value),
                                   xytext=(5, 5), textcoords='offset points',
                                   fontsize=8, ha='left')
                
                plt.title(f'GBM Forecasts 2024-2030 - {gbm_series}', fontsize=14, fontweight='bold')
                plt.xlabel('Year')
                plt.ylabel('Value (Millions)')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plot_num += 1
        
        # 7. Labor Productivity Analysis (only if we have profit results)
        if hasattr(self, 'profit_results') and self.profit_results:
            if 'labor_productivity' in self.profit_results:
                prod_series = self.profit_results['labor_productivity']['series'].dropna()
                if len(prod_series) > 0:
                    plt.subplot(rows, 2, plot_num)
                    plt.plot(prod_series.index, prod_series, 'o-', label='Labor Productivity', 
                            linewidth=2, color='green', markersize=6)
                    plt.title('Labor Productivity Analysis', fontsize=14, fontweight='bold')
                    plt.xlabel('Year')
                    plt.ylabel('Output per Worker')
                    plt.legend()
                    plt.grid(True, alpha=0.3)
                    plot_num += 1
        
        # 8. Labor Cost Ratio Analysis (only if we have profit results)
        if hasattr(self, 'profit_results') and self.profit_results:
            if 'profit_margin' in self.profit_results:
                margin_series = self.profit_results['profit_margin']['series'].dropna()
                if len(margin_series) > 0:
                    plt.subplot(rows, 2, plot_num)
                    plt.plot(margin_series.index, margin_series * 100, 's-', label='Labor Cost Ratio %', 
                            linewidth=2, color='red', markersize=6)
                    plt.title('Labor Cost Ratio Analysis', fontsize=14, fontweight='bold')
                    plt.xlabel('Year')
                    plt.ylabel('Labor Cost as % of Output')
                    plt.legend()
                    plt.grid(True, alpha=0.3)
                    # Add a horizontal line at 0% for reference
                    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
                    plot_num += 1
        
        # 9. Residual Analysis (only if we have ARIMA results)
        if hasattr(self, 'arima_results') and self.arima_results:
            # Find a series with residuals
            residual_series = None
            for series_name in ['RealGrossOutput', 'RealValueAdded', 'Compensation']:
                if series_name in self.arima_results and 'residuals' in self.arima_results[series_name]:
                    residuals = self.arima_results[series_name]['residuals'].dropna()
                    # Remove 2012 data point as it's an outlier
                    residuals_filtered = residuals[residuals.index.year != 2012]
                    if len(residuals_filtered) > 0:
                        residual_series = (series_name, residuals_filtered)
                        break
            
            if residual_series:
                plt.subplot(rows, 2, plot_num)
                series_name, residuals = residual_series
                plt.plot(residuals.index, residuals, 'o-', alpha=0.7, label=f'{series_name} Residuals (2012 excluded)')
                plt.title('ARIMA Model Residuals (Outlier Removed)', fontsize=14, fontweight='bold')
                plt.xlabel('Year')
                plt.ylabel('Residuals')
                plt.axhline(y=0, color='red', linestyle='--', alpha=0.5)
                plt.legend()
                plt.grid(True, alpha=0.3)
                plot_num += 1
        
        # 10. Correlation Heatmap (only if we have at least 2 series)
        if len(available_series) >= 2:
            plt.subplot(rows, 2, plot_num)
            # Create correlation matrix only for available series
            correlation_series = list(available_series.keys())
            correlation_data = self.time_series[correlation_series].corr()
            
            import seaborn as sns
            sns.heatmap(correlation_data, annot=True, cmap='coolwarm', center=0, 
                       square=True, cbar_kws={'label': 'Correlation'})
            plt.title('Correlation Matrix - Available Indicators', fontsize=14, fontweight='bold')
            plot_num += 1
        
        plt.tight_layout()
        plt.savefig('space_economy_comprehensive_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ Visualizations saved as 'space_economy_comprehensive_analysis.png'")
    
    def generate_summary_report(self):
        """Generate a comprehensive summary report"""
        print("\n" + "="*60)
        print("SPACE ECONOMY ANALYSIS SUMMARY REPORT")
        print("="*60)
        
        print(f"\n📊 DATASET OVERVIEW:")
        print(f"   • Time Period: {self.time_series.index[0].year} - {self.time_series.index[-1].year}")
        print(f"   • Variables: {len(self.time_series.columns)}")
        print(f"   • Observations per series: {len(self.time_series)}")
        
        print(f"\n📈 KEY STATISTICS:")
        for col in ['RealGrossOutput', 'Employment', 'Compensation']:
            if col in self.time_series.columns:
                series = self.time_series[col]
                growth = series.pct_change().mean() * 100
                volatility = series.pct_change().std() * 100
                print(f"   • {col}:")
                print(f"     - Average annual growth: {growth:.2f}%")
                print(f"     - Volatility: {volatility:.2f}%")
        
        if hasattr(self, 'arima_results'):
            print(f"\n📊 ARIMA MODELS (2024-2030 FORECASTS):")
            for series_name, results in self.arima_results.items():
                print(f"   • {series_name}: ARIMA{results['order']} (AIC: {results['aic']:.1f})")
                if 'forecast_years' in results:
                    forecast_mean = results['forecast_mean']
                    print(f"     - 2024: {forecast_mean.iloc[0]:,.0f}")
                    print(f"     - 2030: {forecast_mean.iloc[-1]:,.0f}")
                    growth_rate = (forecast_mean.iloc[-1] / results['last_observed_value']) ** (1/7) - 1
                    print(f"     - Implied annual growth (2023-2030): {growth_rate*100:.1f}%")
        
        if hasattr(self, 'gbm_results'):
            print(f"\n📈 GBM MODELS (2024-2030 FORECASTS):")
            for series_name, results in self.gbm_results.items():
                print(f"   • {series_name}:")
                print(f"     - Annual drift (μ): {results['mu']*100:.2f}%")
                print(f"     - Annual volatility (σ): {results['sigma']*100:.2f}%")
                if 'forecast_percentiles' in results:
                    p50_2024 = results['forecast_percentiles']['p50'].iloc[0]
                    p50_2030 = results['forecast_percentiles']['p50'].iloc[-1]
                    print(f"     - 2024 median forecast: {p50_2024:,.0f}")
                    print(f"     - 2030 median forecast: {p50_2030:,.0f}")
        
        if hasattr(self, 'profit_results'):
            print(f"\n💰 PROFITABILITY INSIGHTS:")
            if 'labor_productivity' in self.profit_results:
                avg_growth = self.profit_results['labor_productivity']['avg_growth'] * 100
                print(f"   • Labor productivity growth: {avg_growth:.2f}% annually")
            
            if 'profit_margin' in self.profit_results:
                avg_ratio = self.profit_results['profit_margin']['avg_margin'] * 100
                trend = self.profit_results['profit_margin']['trend'] * 100
                print(f"   • Average labor cost ratio: {avg_ratio:.1f}% of output")
                print(f"   • Labor cost ratio trend: {trend:+.2f}% per year")
        
        print(f"\n🔮 FORECASTING:")
        print(f"   • ARIMA models fitted for key economic indicators")
        print(f"   • GBM models for monetary series with Monte Carlo simulation") 
        print(f"   • 7-year projections: 2024, 2025, 2026, 2027, 2028, 2029, 2030")
        print(f"   • Forecasts start from last observed data point (2023)")
        print(f"   • Confidence intervals provided for uncertainty quantification")
        
        print("\n" + "="*60)

def main():
    """Main analysis pipeline"""
    print("🚀 SPACE ECONOMY DATA ANALYSIS - CDC 2025")
    print("="*50)
    
    # Initialize analyzer
    analyzer = SpaceEconomyAnalyzer()
    
    # Load and process data
    data = analyzer.load_data_from_txt_files()
    if data.empty:
        print("❌ No data loaded. Please check your .txt files.")
        return
    
    print(f"\n📋 Loaded Data Summary:")
    print(data.describe().round(2))
    
    # Create transformations
    transformations = analyzer.create_transformations()
    
    # Statistical analysis
    adf_results = analyzer.stationarity_tests()
    print(f"\n📊 Stationarity Test Results:")
    print(adf_results.round(4))
    
    # Model fitting
    arima_results = analyzer.fit_arima_models()
    gbm_results = analyzer.geometric_brownian_motion()
    profit_results = analyzer.profit_analysis()
    
    # Create visualizations
    analyzer.create_visualizations()
    
    # Generate summary report
    analyzer.generate_summary_report()
    
    print("\n✅ Analysis complete! Check the generated plots and summary above.")

if __name__ == "__main__":
    main()
