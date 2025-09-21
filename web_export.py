# Web Export Module for Interactive Space Economy Visualization
import json
import pandas as pd
from main import SpaceEconomyAnalyzer

class WebDataExporter:
    """Export space economy data for web visualization"""
    
    def __init__(self):
        self.analyzer = SpaceEconomyAnalyzer()
        
    def prepare_data(self):
        """Prepare all data needed for the web interface"""
        # Load the core data
        self.analyzer.load_data_from_txt_files()
        self.analyzer.create_transformations()
        self.analyzer.stationarity_tests()
        self.analyzer.profit_analysis()
        
        # Generate forecasts with extended periods (2024-2030 = 7 years)
        self.analyzer.fit_arima_models(forecast_periods=7)
        self.analyzer.geometric_brownian_motion(n_simulations=1000, forecast_periods=7)
        
        return self.create_web_dataset()
    
    def create_web_dataset(self):
        """Create a comprehensive dataset for web visualization"""
        # Convert timestamps to years for easier handling
        years = [int(ts.year) for ts in self.analyzer.time_series.index]
        
        # Base economic data
        web_data = {
            'years': years,
            'data': {
                'economic_output': {
                    'real_value_added': self.analyzer.time_series['RealValueAdded'].tolist(),
                    'nominal_value_added': self.analyzer.time_series['NominalValueAdded'].tolist(),
                    'real_gross_output': self.analyzer.time_series['RealGrossOutput'].tolist(),
                    'nominal_gross_output': self.analyzer.time_series['NominalGrossOutput'].tolist(),
                },
                'labor_market': {
                    'employment': self.analyzer.time_series['Employment'].tolist(),
                    'compensation': self.analyzer.time_series['Compensation'].tolist(),
                },
                'price_indices': {
                    'value_added_price': self.analyzer.time_series['ValueAddedPriceIndex'].tolist(),
                    'gross_output_price': self.analyzer.time_series['GrossOutputPriceIndex'].tolist(),
                }
            }
        }
        
        # Add growth rates if available
        if hasattr(self.analyzer, 'transformations'):
            growth_data = {}
            for col in self.analyzer.time_series.columns:
                growth_col = f'{col}_growth'
                if growth_col in self.analyzer.transformations.columns:
                    # Skip the first NaN value and align with years[1:]
                    growth_values = self.analyzer.transformations[growth_col].dropna().tolist()
                    growth_data[col.lower()] = growth_values
            
            web_data['growth_rates'] = growth_data
            web_data['growth_years'] = years[1:]  # Growth rates start from second year
        
        # Add productivity analysis if available
        if hasattr(self.analyzer, 'profit_results'):
            productivity_data = {}
            
            # Labor productivity (per worker)
            if 'labor_productivity' in self.analyzer.profit_results:
                prod_data = self.analyzer.profit_results['labor_productivity']
                productivity_data['labor_productivity'] = prod_data['series'].tolist()
                productivity_data['avg_growth'] = prod_data['avg_growth']
                productivity_data['volatility'] = prod_data['volatility']
            
            # Labor cost ratio
            if 'profit_margin' in self.analyzer.profit_results:
                cost_data = self.analyzer.profit_results['profit_margin']
                productivity_data['labor_cost_ratio'] = cost_data['series'].tolist()
                productivity_data['avg_cost_ratio'] = cost_data['avg_margin']
            
            # Average compensation
            if 'avg_compensation' in self.analyzer.profit_results:
                comp_data = self.analyzer.profit_results['avg_compensation']
                productivity_data['avg_compensation'] = comp_data['series'].tolist()
            
            web_data['productivity'] = productivity_data
        
        # Add metadata
        web_data['metadata'] = {
            'title': 'U.S. Space Economy Analysis (2012-2023)',
            'total_years': len(years),
            'variables': list(self.analyzer.time_series.columns),
            'units': {
                'economic_output': 'Millions of USD',
                'employment': 'Thousands of workers',
                'compensation': 'Millions of USD',
                'price_indices': 'Index (base year = 100)',
                'growth_rates': 'Percentage change',
                'productivity': 'Various units'
            }
        }
        
        # Add forecasting data
        forecast_years = list(range(2024, 2031))  # 2024-2030
        all_years = years + forecast_years
        
        web_data['forecasts'] = self.create_forecast_data(all_years)
        web_data['all_years'] = all_years
        web_data['forecast_years'] = forecast_years
        
        return web_data
    
    def create_forecast_data(self, all_years):
        """Create forecasting data for ARIMA and GBM models"""
        forecast_data = {
            'arima': {},
            'gbm': {},
            'years': all_years
        }
        
        # Get ARIMA forecasts if available
        if hasattr(self.analyzer, 'arima_results'):
            for var, results in self.analyzer.arima_results.items():
                if 'forecast_mean' in results:
                    # Combine historical data with forecasts
                    historical = self.analyzer.time_series[var].tolist()
                    forecast_values = results['forecast_mean'].tolist()
                    combined_data = historical + forecast_values
                    
                    # Get confidence intervals if available
                    conf_intervals = results.get('forecast_ci', None)
                    conf_lower = []
                    conf_upper = []
                    if conf_intervals is not None:
                        conf_lower = conf_intervals.iloc[:, 0].tolist()
                        conf_upper = conf_intervals.iloc[:, 1].tolist()
                    
                    forecast_data['arima'][var.lower()] = {
                        'data': combined_data,
                        'confidence_lower': historical + ([None] * len(historical) + conf_lower)[-len(all_years):],
                        'confidence_upper': historical + ([None] * len(historical) + conf_upper)[-len(all_years):],
                        'historical_end_index': len(historical) - 1
                    }
        
        # Get GBM simulations if available
        if hasattr(self.analyzer, 'gbm_results'):
            for var, results in self.analyzer.gbm_results.items():
                if 'forecast_percentiles' in results:
                    # Use median forecast for main prediction
                    historical = self.analyzer.time_series[var].tolist()
                    percentiles = results['forecast_percentiles']
                    
                    # Extract median (p50)
                    median_forecast = percentiles['p50'].tolist()
                    combined_data = historical + median_forecast
                    
                    # Get uncertainty bands
                    p25 = percentiles['p25'].tolist()
                    p75 = percentiles['p75'].tolist()
                    
                    forecast_data['gbm'][var.lower()] = {
                        'data': combined_data,
                        'percentile_25': historical + ([None] * len(historical) + p25)[-len(all_years):],
                        'percentile_75': historical + ([None] * len(historical) + p75)[-len(all_years):],
                        'historical_end_index': len(historical) - 1
                    }
        
        return forecast_data
    
    def export_to_json(self, filename='space_economy_data.json'):
        """Export data to JSON file for web consumption"""
        data = self.prepare_data()
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        print(f"✅ Exported web data to {filename}")
        return data

if __name__ == "__main__":
    exporter = WebDataExporter()
    data = exporter.export_to_json()
    print(f"Exported data for {len(data['years'])} years with {len(data['data'])} categories")