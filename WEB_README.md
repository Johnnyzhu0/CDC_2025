# Interactive Space Economy Web Interface

This web interface provides an interactive visualization of U.S. Space Economy data with **clear separation between historical analysis (2012-2023) and predictive forecasting (2012-2030)**.

## Features

### 🎮 Interactive Controls
- **Year Slider**: Historical data from 2012 to 2023 for interactive exploration
- **Chart Selector Dropdown**: Choose between "All Charts" view or focus on individual charts
- **Play Animation**: Automatically cycles through historical years to show trends
- **Speed Control**: Adjust animation speed (Slow/Normal/Fast)
- **Reset Button**: Jump back to the beginning

### 📊 Dynamic Visualizations

#### Historical Analysis Section (2012-2023) - Interactive
1. **Economic Output Indicators**: Real/Nominal Value Added and Gross Output
2. **Labor Market Indicators**: Employment and Compensation trends
3. **Growth Rates**: Year-over-year percentage changes
4. **Labor Productivity Analysis**: Efficiency metrics per worker

#### Predictive Models Section (2012-2030) - Static Display
1. **ARIMA Forecasts**: Statistical predictions for Economic Indicators (Real Value Added, Real Gross Output, Employment)
2. **GBM Monte Carlo Simulations**: Stochastic forecasts for Nominal indicators (Nominal Value Added, Nominal Gross Output)

### 📈 Key Features
- **Historical Interactivity**: Use slider and animation to explore 2012-2023 data
- **Forecast Overview**: Static display of complete ARIMA and GBM forecasts through 2030
- **Chart Focus Mode**: Dropdown to view one historical chart at a time for detailed analysis
- **Model Separation**: ARIMA for conservative statistical forecasts, GBM for stochastic projections

### 📈 Real-time Stats Display
- Current year's key metrics
- Real Value Added, Employment, Compensation
- Productivity per worker calculations

## How to Use

1. **Start the local server**:
   ```bash
   cd /Users/inori/IdeaProjects/CDC_2025
   python3 -m http.server 8002
   ```

2. **Open in browser**: Navigate to `http://localhost:8002`

3. **Explore the data**:
   - **Historical Analysis**: Use the year slider (2012-2023) to interactively explore actual trends
   - **Chart Selection**: Choose specific charts from the dropdown for focused analysis
   - **Animation**: Click "Play Animation" to watch the historical timeline unfold
   - **Forecasting**: View complete ARIMA and GBM predictions in the dedicated forecasting section

4. **Navigation Tips**:
   - **All Charts**: Grid view showing all historical visualizations simultaneously
   - **Individual Focus**: Select specific charts for larger, detailed view
   - **Static Forecasts**: Forecasting section shows complete predictions (not interactive)

## Technical Implementation

### Data Pipeline
- `web_export.py`: Exports analysis results to JSON format
- `space_economy_data.json`: Contains all economic data and calculations
- Real-time chart updates using Chart.js library

### Interactive Features
- **Smooth Animations**: Charts update without flickering
- **Responsive Design**: Works on desktop and tablet devices
- **Dual-axis Charts**: Labor market shows employment and compensation simultaneously
- **Progressive Data Display**: Only shows data up to selected year

## Key Insights to Explore

### Historical Patterns (2012-2023) - Interactive
- **2020 Impact**: Use the slider to see the dip in economic indicators during COVID-19
- **Recovery Pattern**: Animate through 2021-2023 to watch the strong growth resumption
- **Employment Trends**: Observe relatively stable workforce with productivity gains
- **Compensation Volatility**: Notice significant compensation increases reflecting industry value

### Predictive Insights (2024-2030) - Static Overview
- **ARIMA Forecasts**: Conservative statistical predictions based on historical patterns
  - Real Value Added: Gradual decline projected
  - Real Gross Output: Steady growth continuation
  - Employment: Stable workforce levels maintained
- **GBM Simulations**: Stochastic modeling with natural market volatility
  - Nominal Value Added: Continued growth with uncertainty bands
  - Nominal Gross Output: Strong expansion trajectory
- **Model Differences**: Compare conservative ARIMA vs optimistic GBM projections

## Files Structure

```
├── index.html              # Main web interface
├── space_economy_app.js     # Interactive JavaScript application
├── space_economy_data.json  # Exported economic data
├── web_export.py           # Data export utility
└── main.py                 # Original analysis code
```

## Browser Compatibility

- Chrome/Edge: Full support
- Firefox: Full support  
- Safari: Full support
- Mobile browsers: Responsive design

The interface leverages Chart.js for smooth, professional visualizations and provides an engaging way to explore the space economy dataset interactively.