# Space Economy Data Analysis - Carolina Data Challenge 2025
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller, grangercausalitytests
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
                years = [col.strip() for col in header_line.split() if col.strip().isdigit()]
                
                # Find the total/aggregate row (usually first data row)
                total_row = None
                for line in data_lines[1:]:
                    parts = line.split()
                    if len(parts) >= len(years) + 2:  # Index + description + data
                        # Extract numeric values
                        numeric_data = []
                        for part in parts[-len(years):]:  # Take last N values where N = number of years
                            try:
                                if part != '…' and part != 'nan':
                                    numeric_data.append(float(part))
                                else:
                                    numeric_data.append(np.nan)
                            except:
                                numeric_data.append(np.nan)
                        
                        if len(numeric_data) == len(years) and not all(pd.isna(numeric_data)):
                            total_row = numeric_data
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
    
    def granger_causality_analysis(self):
        """Perform Granger causality tests"""
        print("\nPerforming Granger causality tests...")
        
        # Define key relationships to test
        test_pairs = [
            ('RealGrossOutput_dlog', 'Employment_dlog'),
            ('RealGrossOutput_dlog', 'Compensation_dlog'), 
            ('RealValueAdded_dlog', 'Employment_dlog'),
            ('Employment_dlog', 'Compensation_dlog'),
            ('NominalGrossOutput_dlog', 'Compensation_dlog'),
            ('RealValueAdded_dlog', 'RealGrossOutput_dlog')
        ]
        
        granger_results = []
        
        for x_var, y_var in test_pairs:
            if x_var in self.transformations.columns and y_var in self.transformations.columns:
                data = self.transformations[[y_var, x_var]].dropna()
                
                if len(data) >= 6:  # Need sufficient observations
                    maxlag = min(2, len(data)//3)  # Conservative lag selection
                    try:
                        test_result = grangercausalitytests(data, maxlag=maxlag, verbose=False)
                        
                        for lag in test_result.keys():
                            f_stat = test_result[lag][0]['ssr_ftest'][0]
                            p_value = test_result[lag][0]['ssr_ftest'][1]
                            
                            granger_results.append({
                                'X_causes_Y': f'{x_var} → {y_var}',
                                'Lag': lag,
                                'F_statistic': f_stat,
                                'p_value': p_value,
                                'Significant': p_value < 0.1  # Using 10% significance
                            })
                    except Exception as e:
                        granger_results.append({
                            'X_causes_Y': f'{x_var} → {y_var}',
                            'Lag': 1,
                            'F_statistic': np.nan,
                            'p_value': np.nan,
                            'Significant': False,
                            'Error': str(e)
                        })
        
        self.granger_results = pd.DataFrame(granger_results)
        return self.granger_results
    
    def fit_arima_models(self, forecast_periods=5):
        """Fit ARIMA models for forecasting"""
        print(f"\nFitting ARIMA models (forecasting {forecast_periods} periods)...")
        
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
                        # Generate forecasts
                        try:
                            forecast = best_model.get_forecast(steps=forecast_periods)
                            forecast_index = pd.date_range(
                                start=series.index[-1] + pd.DateOffset(years=1),
                                periods=forecast_periods,
                                freq='AS'
                            )
                            
                            arima_results[series_name] = {
                                'model': best_model,
                                'order': best_order,
                                'aic': best_aic,
                                'forecast_mean': pd.Series(forecast.predicted_mean, index=forecast_index),
                                'forecast_ci': forecast.conf_int(),
                                'fitted_values': best_model.fittedvalues,
                                'residuals': best_model.resid
                            }
                        except Exception as e:
                            print(f"    ❌ Forecast error for {series_name}: {e}")
                    else:
                        print(f"    ❌ Could not fit ARIMA for {series_name}")
        
        self.arima_results = arima_results
        return arima_results
    
    def geometric_brownian_motion(self, n_simulations=1000, forecast_periods=5):
        """Fit Geometric Brownian Motion model"""
        print(f"\nFitting Geometric Brownian Motion (GBM) models...")
        
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
                    S0 = series.iloc[-1]  # Last observed value
                    
                    # Generate forecast paths
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
                    
                    forecast_index = pd.date_range(
                        start=series.index[-1] + pd.DateOffset(years=1),
                        periods=forecast_periods,
                        freq='AS'
                    )
                    
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
                        'last_value': S0
                    }
        
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
        
        # Profit margin proxy (if we have both output and compensation)
        if 'RealGrossOutput' in self.time_series.columns and 'Compensation' in self.time_series.columns:
            # Simple profit margin approximation
            profit_margin = (self.time_series['RealGrossOutput'] - self.time_series['Compensation']) / self.time_series['RealGrossOutput']
            profit_margin.name = 'ProfitMargin'
            
            margin_change = profit_margin.diff()
            
            profit_results['profit_margin'] = {
                'series': profit_margin,
                'change': margin_change,
                'avg_margin': profit_margin.mean(),
                'trend': stats.linregress(range(len(profit_margin.dropna())), profit_margin.dropna()).slope
            }
        
        self.profit_results = profit_results
        return profit_results
    
    def create_visualizations(self):
        """Create comprehensive visualizations"""
        print("\nCreating visualizations...")
        
        # Set up the plotting layout
        fig = plt.figure(figsize=(20, 24))
        
        # 1. Time series overview
        plt.subplot(4, 2, 1)
        for col in self.time_series.columns:
            if col in ['RealGrossOutput', 'Employment', 'Compensation', 'RealValueAdded']:
                plt.plot(self.time_series.index, self.time_series[col], marker='o', label=col, linewidth=2)
        plt.title('Key Space Economy Indicators (2012-2023)', fontsize=14, fontweight='bold')
        plt.xlabel('Year')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 2. Growth rates
        plt.subplot(4, 2, 2)
        for col in ['RealGrossOutput', 'Employment', 'Compensation']:
            if f'{col}_pct_change' in self.transformations.columns:
                growth = self.transformations[f'{col}_pct_change'].dropna() * 100
                plt.plot(growth.index, growth, marker='s', label=f'{col} Growth %', linewidth=2)
        plt.title('Annual Growth Rates', fontsize=14, fontweight='bold')
        plt.xlabel('Year')
        plt.ylabel('Growth Rate (%)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        
        # 3. ARIMA Forecasts
        plt.subplot(4, 2, 3)
        if hasattr(self, 'arima_results'):
            for series_name, results in self.arima_results.items():
                if series_name == 'RealGrossOutput':  # Focus on main series
                    historical = self.time_series[series_name]
                    forecast = results['forecast_mean']
                    
                    plt.plot(historical.index, historical, 'o-', label=f'{series_name} (Historical)', linewidth=2)
                    plt.plot(forecast.index, forecast, 's--', label=f'{series_name} (ARIMA Forecast)', linewidth=2)
                    
                    # Add confidence interval
                    ci = results['forecast_ci']
                    plt.fill_between(forecast.index, ci.iloc[:, 0], ci.iloc[:, 1], alpha=0.2)
        
        plt.title('ARIMA Forecasts - Real Gross Output', fontsize=14, fontweight='bold')
        plt.xlabel('Year')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 4. GBM Simulation Paths
        plt.subplot(4, 2, 4)
        if hasattr(self, 'gbm_results'):
            for series_name, results in self.gbm_results.items():
                if series_name == 'NominalGrossOutput':  # Focus on main monetary series
                    historical = self.time_series[series_name]
                    percentiles = results['forecast_percentiles']
                    
                    plt.plot(historical.index, historical, 'o-', label=f'{series_name} (Historical)', linewidth=2)
                    
                    # Plot percentile bands
                    plt.plot(percentiles.index, percentiles['p50'], 's-', label='GBM Median', linewidth=2)
                    plt.fill_between(percentiles.index, percentiles['p5'], percentiles['p95'], 
                                   alpha=0.2, label='90% Confidence Band')
                    plt.fill_between(percentiles.index, percentiles['p25'], percentiles['p75'], 
                                   alpha=0.3, label='50% Confidence Band')
        
        plt.title('GBM Forecasts - Nominal Gross Output', fontsize=14, fontweight='bold')
        plt.xlabel('Year')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 5. Granger Causality Results
        plt.subplot(4, 2, 5)
        if hasattr(self, 'granger_results') and not self.granger_results.empty:
            significant_results = self.granger_results[self.granger_results['Significant'] == True]
            
            if not significant_results.empty:
                relationships = significant_results['X_causes_Y'].str.replace('_dlog', '').str.replace(' → ', ' → ')
                p_values = significant_results['p_value']
                
                bars = plt.barh(range(len(relationships)), -np.log10(p_values), color='skyblue')
                plt.yticks(range(len(relationships)), relationships)
                plt.xlabel('-log10(p-value)')
                plt.title('Significant Granger Causality Results', fontsize=14, fontweight='bold')
                plt.axvline(x=-np.log10(0.05), color='red', linestyle='--', label='5% significance')
                plt.axvline(x=-np.log10(0.1), color='orange', linestyle='--', label='10% significance')
                plt.legend()
                
                # Add value labels on bars
                for i, bar in enumerate(bars):
                    width = bar.get_width()
                    plt.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                            f'{p_values.iloc[i]:.3f}', ha='left', va='center')
        
        plt.grid(True, alpha=0.3)
        
        # 6. Profit Analysis
        plt.subplot(4, 2, 6)
        if hasattr(self, 'profit_results'):
            if 'labor_productivity' in self.profit_results:
                prod_series = self.profit_results['labor_productivity']['series']
                plt.plot(prod_series.index, prod_series, 'o-', label='Labor Productivity', linewidth=2, color='green')
                
            if 'profit_margin' in self.profit_results:
                margin_series = self.profit_results['profit_margin']['series']
                plt.plot(margin_series.index, margin_series * 100, 's-', label='Profit Margin %', linewidth=2, color='red')
                
        plt.title('Productivity and Profitability Metrics', fontsize=14, fontweight='bold')
        plt.xlabel('Year')
        plt.ylabel('Value / Percentage')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 7. Residual Analysis (ARIMA)
        plt.subplot(4, 2, 7)
        if hasattr(self, 'arima_results'):
            for series_name, results in self.arima_results.items():
                if series_name == 'RealGrossOutput':
                    residuals = results['residuals']
                    plt.plot(residuals.index, residuals, 'o-', alpha=0.7, label=f'{series_name} Residuals')
                    break
        
        plt.title('ARIMA Model Residuals', fontsize=14, fontweight='bold')
        plt.xlabel('Year')
        plt.ylabel('Residuals')
        plt.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        plt.grid(True, alpha=0.3)
        
        # 8. Correlation Heatmap
        plt.subplot(4, 2, 8)
        correlation_data = self.time_series[['RealGrossOutput', 'Employment', 'Compensation', 'RealValueAdded']].corr()
        sns.heatmap(correlation_data, annot=True, cmap='coolwarm', center=0, 
                   square=True, cbar_kws={'label': 'Correlation'})
        plt.title('Correlation Matrix - Key Indicators', fontsize=14, fontweight='bold')
        
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
        
        if hasattr(self, 'granger_results'):
            sig_results = self.granger_results[self.granger_results['Significant'] == True]
            print(f"\n🔗 GRANGER CAUSALITY:")
            if not sig_results.empty:
                print(f"   • Found {len(sig_results)} significant causal relationships")
                for _, row in sig_results.iterrows():
                    print(f"     - {row['X_causes_Y']} (p-value: {row['p_value']:.3f})")
            else:
                print("   • No significant causal relationships found at 10% level")
        
        if hasattr(self, 'arima_results'):
            print(f"\n📊 ARIMA MODELS:")
            for series_name, results in self.arima_results.items():
                print(f"   • {series_name}: ARIMA{results['order']} (AIC: {results['aic']:.1f})")
        
        if hasattr(self, 'profit_results'):
            print(f"\n💰 PROFITABILITY INSIGHTS:")
            if 'labor_productivity' in self.profit_results:
                avg_growth = self.profit_results['labor_productivity']['avg_growth'] * 100
                print(f"   • Labor productivity growth: {avg_growth:.2f}% annually")
            
            if 'profit_margin' in self.profit_results:
                avg_margin = self.profit_results['profit_margin']['avg_margin'] * 100
                trend = self.profit_results['profit_margin']['trend'] * 100
                print(f"   • Average profit margin: {avg_margin:.1f}%")
                print(f"   • Profit margin trend: {trend:+.2f}% per year")
        
        print(f"\n🔮 FORECASTING:")
        print(f"   • ARIMA models fitted for key economic indicators")
        print(f"   • GBM models for monetary series with Monte Carlo simulation") 
        print(f"   • 5-year forward projections generated")
        
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
    
    granger_results = analyzer.granger_causality_analysis()
    if not granger_results.empty:
        print(f"\n🔗 Granger Causality Results:")
        print(granger_results.round(4))
    
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
