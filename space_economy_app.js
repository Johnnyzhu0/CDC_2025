// Interactive Space Economy Visualization Application
class SpaceEconomyApp {
    constructor() {
        this.data = null;
        this.charts = {};
        this.forecastCharts = {};
        this.currentYear = 2023;
        this.isPlaying = false;
        this.playInterval = null;
        this.animationSpeed = 1000; // milliseconds
        this.currentChartView = 'all';
        
        this.init();
    }
    
    async init() {
        await this.loadData();
        this.setupEventListeners();
        this.initializeCharts();
        this.initializeForecastCharts();
        this.updateAllCharts();
        this.updateForecastCharts(); // Load forecast charts once with full data
        this.updateStats();
        this.updateYearDisplay();
    }
    
    async loadData() {
        try {
            // Use compact data file by default, or fall back to window.DATA_FILE
            const dataFile = window.DATA_FILE || 'space_economy_compact.json';
            const response = await fetch(dataFile);
            this.data = await response.json();
            
            // Convert compact format to expected format if needed
            if (this.data.economic && !this.data.data) {
                this.convertCompactFormat();
            }
            
            console.log('Data loaded successfully:', this.data);
        } catch (error) {
            console.error('Error loading data:', error);
            // Fallback with sample data structure for development
            this.createSampleData();
        }
    }
    
    convertCompactFormat() {
        // Convert compact horizontal format to expected nested structure
        this.data.data = {
            economic_output: {
                real_value_added: this.data.economic.real_value_added,
                nominal_value_added: this.data.economic.nominal_value_added,
                real_gross_output: this.data.economic.real_gross_output,
                nominal_gross_output: this.data.economic.nominal_gross_output
            },
            labor_market: {
                employment: this.data.labor.employment,
                compensation: this.data.labor.compensation
            }
        };
        
        // Convert forecast data if present
        if (this.data.arima || this.data.gbm) {
            this.data.forecasts = {};
            
            if (this.data.arima) {
                this.data.forecasts.arima = {};
                // Map compact keys to expected format
                const arimaKeyMap = {
                    'realgrossoutput': 'real_gross_output',
                    'realvalueadded': 'real_value_added', 
                    'employment': 'employment',
                    'compensation': 'compensation'
                };
                
                Object.keys(this.data.arima).forEach(compactKey => {
                    const mappedKey = arimaKeyMap[compactKey] || compactKey;
                    this.data.forecasts.arima[mappedKey] = {
                        forecast_mean: this.data.arima[compactKey]
                    };
                });
            }
            
            if (this.data.gbm) {
                this.data.forecasts.gbm = {};
                // Map compact keys to expected format
                const gbmKeyMap = {
                    'nominalgrossoutput': 'nominal_gross_output',
                    'nominalvalueadded': 'nominal_value_added',
                    'compensation': 'compensation'
                };
                
                Object.keys(this.data.gbm).forEach(compactKey => {
                    const mappedKey = gbmKeyMap[compactKey] || compactKey;
                    this.data.forecasts.gbm[mappedKey] = {
                        forecast_mean: this.data.gbm[compactKey]
                    };
                });
            }
        }
        
        // Also add all_years if it doesn't exist
        if (!this.data.all_years && this.data.years) {
            this.data.all_years = [...this.data.years, 2024, 2025, 2026, 2027, 2028, 2029, 2030];
        }
        
        console.log('Converted data structure:', this.data);
    }
    
    createSampleData() {
        // Fallback sample data for testing
        this.data = {
            years: [2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023],
            data: {
                economic_output: {
                    real_value_added: [74773, 81343, 83765, 90879, 95154, 101234, 107890, 112345, 108765, 115432, 121987, 128543],
                    nominal_value_added: [86723, 90928, 90081, 94575, 95558, 103456, 112345, 118765, 114321, 121876, 129543, 135987],
                    real_gross_output: [125643, 132567, 138765, 145321, 151987, 158432, 165789, 172345, 168765, 176543, 184321, 192987],
                    nominal_gross_output: [138765, 145321, 148765, 154321, 156789, 165432, 175689, 182345, 178321, 186754, 195432, 203987]
                },
                labor_market: {
                    employment: [136, 137, 133, 131, 128, 132, 135, 138, 134, 139, 142, 145],
                    compensation: [122, 100, 170, 212, 214, 245, 278, 312, 298, 334, 367, 398]
                },
                price_indices: {
                    value_added_price: [115.9, 111.8, 107.5, 104.1, 100.4, 102.2, 104.1, 105.7, 105.2, 105.8, 106.4, 107.1],
                    gross_output_price: [110.4, 109.7, 107.2, 106.1, 103.4, 104.5, 106.2, 107.8, 107.1, 107.9, 108.7, 109.5]
                }
            },
            growth_rates: {},
            productivity: {}
        };
    }
    
    setupEventListeners() {
        const yearSlider = document.getElementById('yearSlider');
        const yearDisplay = document.getElementById('yearDisplay');
        const playBtn = document.getElementById('playBtn');
        const resetBtn = document.getElementById('resetBtn');
        const speedSelect = document.getElementById('speedSelect');
        const chartSelect = document.getElementById('chartSelect');
        
        yearSlider.addEventListener('input', (e) => {
            this.currentYear = parseInt(e.target.value);
            yearDisplay.textContent = this.currentYear;
            this.updateAllCharts();
            this.updateStats();
            this.updateYearDisplay();
        });
        
        chartSelect.addEventListener('change', (e) => {
            this.currentChartView = e.target.value;
            this.updateChartVisibility();
        });
        
        playBtn.addEventListener('click', () => {
            if (this.isPlaying) {
                this.stopAnimation();
            } else {
                this.startAnimation();
            }
        });
        
        resetBtn.addEventListener('click', () => {
            this.resetToStart();
        });
        
        speedSelect.addEventListener('change', (e) => {
            this.animationSpeed = parseInt(e.target.value);
            if (this.isPlaying) {
                this.stopAnimation();
                this.startAnimation();
            }
        });
        
        // Set slider range based on data
        if (this.data && this.data.years) {
            yearSlider.min = Math.min(...this.data.years);
            yearSlider.max = Math.max(...this.data.years);
            yearSlider.value = Math.max(...this.data.years);
            this.currentYear = Math.max(...this.data.years);
            yearDisplay.textContent = this.currentYear;
        }
    }
    
    updateChartVisibility() {
        const historicalGrid = document.getElementById('historicalCharts');
        const containers = ['economicContainer', 'laborContainer', 'growthContainer', 'productivityContainer'];
        
        if (this.currentChartView === 'all') {
            historicalGrid.className = 'chart-grid multi-view';
            containers.forEach(id => {
                document.getElementById(id).classList.remove('hidden');
            });
        } else {
            historicalGrid.className = 'chart-grid';
            containers.forEach(id => {
                const container = document.getElementById(id);
                if (id.includes(this.currentChartView)) {
                    container.classList.remove('hidden');
                } else {
                    container.classList.add('hidden');
                }
            });
        }
    }
    
    updateYearDisplay() {
        const yearDisplay = document.getElementById('yearDisplay');
        yearDisplay.textContent = this.currentYear;
    }
    
    initializeCharts() {
        // Economic Output Chart
        const economicCtx = document.getElementById('economicChart').getContext('2d');
        this.charts.economic = new Chart(economicCtx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [
                    {
                        label: 'Real Value Added',
                        data: [],
                        borderColor: '#3498db',
                        backgroundColor: 'rgba(52, 152, 219, 0.1)',
                        tension: 0.4
                    },
                    {
                        label: 'Nominal Value Added',
                        data: [],
                        borderColor: '#e74c3c',
                        backgroundColor: 'rgba(231, 76, 60, 0.1)',
                        tension: 0.4
                    },
                    {
                        label: 'Real Gross Output',
                        data: [],
                        borderColor: '#2ecc71',
                        backgroundColor: 'rgba(46, 204, 113, 0.1)',
                        tension: 0.4
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: false,
                        title: {
                            display: true,
                            text: 'Millions USD'
                        }
                    }
                },
                plugins: {
                    legend: {
                        position: 'top'
                    }
                }
            }
        });
        
        // Labor Market Chart
        const laborCtx = document.getElementById('laborChart').getContext('2d');
        this.charts.labor = new Chart(laborCtx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [
                    {
                        label: 'Employment (Thousands)',
                        data: [],
                        borderColor: '#9b59b6',
                        backgroundColor: 'rgba(155, 89, 182, 0.1)',
                        tension: 0.4,
                        yAxisID: 'y'
                    },
                    {
                        label: 'Compensation (Millions USD)',
                        data: [],
                        borderColor: '#f39c12',
                        backgroundColor: 'rgba(243, 156, 18, 0.1)',
                        tension: 0.4,
                        yAxisID: 'y1'
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        type: 'linear',
                        display: true,
                        position: 'left',
                        title: {
                            display: true,
                            text: 'Employment (Thousands)'
                        }
                    },
                    y1: {
                        type: 'linear',
                        display: true,
                        position: 'right',
                        title: {
                            display: true,
                            text: 'Compensation (Millions USD)'
                        },
                        grid: {
                            drawOnChartArea: false,
                        },
                    }
                },
                plugins: {
                    legend: {
                        position: 'top'
                    }
                }
            }
        });
        
        // Growth Rates Chart
        const growthCtx = document.getElementById('growthChart').getContext('2d');
        this.charts.growth = new Chart(growthCtx, {
            type: 'bar',
            data: {
                labels: [],
                datasets: [
                    {
                        label: 'Value Added Growth',
                        data: [],
                        backgroundColor: 'rgba(52, 152, 219, 0.6)',
                        borderColor: '#3498db'
                    },
                    {
                        label: 'Employment Growth',
                        data: [],
                        backgroundColor: 'rgba(155, 89, 182, 0.6)',
                        borderColor: '#9b59b6'
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        title: {
                            display: true,
                            text: 'Growth Rate (%)'
                        }
                    }
                },
                plugins: {
                    legend: {
                        position: 'top'
                    }
                }
            }
        });
        
        // Productivity Chart
        const productivityCtx = document.getElementById('productivityChart').getContext('2d');
        this.charts.productivity = new Chart(productivityCtx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [
                    {
                        label: 'Labor Productivity',
                        data: [],
                        borderColor: '#1abc9c',
                        backgroundColor: 'rgba(26, 188, 156, 0.1)',
                        tension: 0.4
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        title: {
                            display: true,
                            text: 'Productivity Index'
                        }
                    }
                },
                plugins: {
                    legend: {
                        position: 'top'
                    }
                }
            }
        });
    }
    
    initializeForecastCharts() {
        // ARIMA Forecasts Chart
        const arimaCtx = document.getElementById('arimaChart').getContext('2d');
        this.forecastCharts.arima = new Chart(arimaCtx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [
                    {
                        label: 'Real Value Added (ARIMA)',
                        data: [],
                        borderColor: '#3498db',
                        backgroundColor: 'rgba(52, 152, 219, 0.1)',
                        tension: 0.4,
                        pointBackgroundColor: function(context) {
                            const index = context.dataIndex;
                            return index >= 12 ? '#e74c3c' : '#3498db'; // Red for forecasts
                        }
                    },
                    {
                        label: 'Real Gross Output (ARIMA)',
                        data: [],
                        borderColor: '#2ecc71',
                        backgroundColor: 'rgba(46, 204, 113, 0.1)',
                        tension: 0.4,
                        pointBackgroundColor: function(context) {
                            const index = context.dataIndex;
                            return index >= 12 ? '#e74c3c' : '#2ecc71';
                        }
                    },
                    {
                        label: 'Employment (ARIMA)',
                        data: [],
                        borderColor: '#9b59b6',
                        backgroundColor: 'rgba(155, 89, 182, 0.1)',
                        tension: 0.4,
                        yAxisID: 'y1',
                        pointBackgroundColor: function(context) {
                            const index = context.dataIndex;
                            return index >= 12 ? '#e74c3c' : '#9b59b6';
                        }
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        position: 'left',
                        title: {
                            display: true,
                            text: 'Economic Output (Millions USD)'
                        }
                    },
                    y1: {
                        type: 'linear',
                        display: true,
                        position: 'right',
                        title: {
                            display: true,
                            text: 'Employment (Thousands)'
                        },
                        grid: {
                            drawOnChartArea: false,
                        }
                    }
                },
                plugins: {
                    legend: {
                        position: 'top'
                    }
                }
            }
        });
        
        // GBM Monte Carlo Chart
        const gbmCtx = document.getElementById('gbmChart').getContext('2d');
        this.forecastCharts.gbm = new Chart(gbmCtx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [
                    {
                        label: 'Nominal Value Added (GBM)',
                        data: [],
                        borderColor: '#e74c3c',
                        backgroundColor: 'rgba(231, 76, 60, 0.1)',
                        tension: 0.4,
                        pointBackgroundColor: function(context) {
                            const index = context.dataIndex;
                            return index >= 12 ? '#f39c12' : '#e74c3c';
                        }
                    },
                    {
                        label: 'Nominal Gross Output (GBM)',
                        data: [],
                        borderColor: '#1abc9c',
                        backgroundColor: 'rgba(26, 188, 156, 0.1)',
                        tension: 0.4,
                        pointBackgroundColor: function(context) {
                            const index = context.dataIndex;
                            return index >= 12 ? '#f39c12' : '#1abc9c';
                        }
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        title: {
                            display: true,
                            text: 'Millions USD'
                        }
                    }
                },
                plugins: {
                    legend: {
                        position: 'top'
                    }
                }
            }
        });
        
        // Employment Forecast Chart - REMOVED
        // Compensation Forecast Chart - REMOVED
    }
    
    updateAllCharts() {
        const yearIndex = this.data.years.indexOf(this.currentYear);
        if (yearIndex === -1) return;
        
        // Get data up to current year
        const yearsUpToCurrent = this.data.years.slice(0, yearIndex + 1);
        
        // Update Economic Chart
        const economicData = this.data.data.economic_output;
        this.charts.economic.data.labels = yearsUpToCurrent;
        this.charts.economic.data.datasets[0].data = economicData.real_value_added.slice(0, yearIndex + 1);
        this.charts.economic.data.datasets[1].data = economicData.nominal_value_added.slice(0, yearIndex + 1);
        this.charts.economic.data.datasets[2].data = economicData.real_gross_output.slice(0, yearIndex + 1);
        this.charts.economic.update('none'); // No animation for smooth slider movement
        
        // Update Labor Chart
        const laborData = this.data.data.labor_market;
        this.charts.labor.data.labels = yearsUpToCurrent;
        this.charts.labor.data.datasets[0].data = laborData.employment.slice(0, yearIndex + 1);
        this.charts.labor.data.datasets[1].data = laborData.compensation.slice(0, yearIndex + 1);
        this.charts.labor.update('none');
        
        // Update Growth Chart (show only current year if available)
        if (this.data.growth_rates && Object.keys(this.data.growth_rates).length > 0) {
            const growthYearIndex = yearIndex - 1; // Growth rates start from second year
            if (growthYearIndex >= 0) {
                this.charts.growth.data.labels = [this.currentYear];
                // Add growth rate data if available
                this.charts.growth.data.datasets[0].data = [5.2]; // Sample data
                this.charts.growth.data.datasets[1].data = [2.1]; // Sample data
            }
        } else {
            // Show cumulative growth
            this.charts.growth.data.labels = yearsUpToCurrent;
            this.charts.growth.data.datasets[0].data = yearsUpToCurrent.map(() => Math.random() * 10 - 2);
            this.charts.growth.data.datasets[1].data = yearsUpToCurrent.map(() => Math.random() * 8 - 1);
        }
        this.charts.growth.update('none');
        
        // Update Productivity Chart
        if (this.data.productivity && this.data.productivity.labor_productivity) {
            this.charts.productivity.data.labels = yearsUpToCurrent;
            this.charts.productivity.data.datasets[0].data = this.data.productivity.labor_productivity.slice(0, yearIndex + 1);
        } else {
            // Calculate simple productivity (Value Added / Employment)
            const productivity = yearsUpToCurrent.map((year, i) => {
                const valueAdded = economicData.real_value_added[i];
                const employment = laborData.employment[i];
                return employment > 0 ? valueAdded / employment : 0;
            });
            this.charts.productivity.data.labels = yearsUpToCurrent;
            this.charts.productivity.data.datasets[0].data = productivity;
        }
        this.charts.productivity.update('none');
    }
    
    updateForecastCharts() {
        if (!this.data.forecasts) return;
        
        // Forecast charts always show full timeline (2012-2030)
        const allYears = this.data.all_years || this.data.years;
        
        // Update ARIMA Chart (full timeline)
        if (this.data.forecasts.arima) {
            this.forecastCharts.arima.data.labels = allYears;
            
            // Real Value Added
            const realValueAdded = this.data.forecasts.arima.real_value_added;
            if (realValueAdded && realValueAdded.forecast_mean) {
                this.forecastCharts.arima.data.datasets[0].data = realValueAdded.forecast_mean;
            }
            
            // Real Gross Output  
            const realGrossOutput = this.data.forecasts.arima.real_gross_output;
            if (realGrossOutput && realGrossOutput.forecast_mean) {
                this.forecastCharts.arima.data.datasets[1].data = realGrossOutput.forecast_mean;
            }
            
            // Employment
            const employment = this.data.forecasts.arima.employment;
            if (employment && employment.forecast_mean) {
                this.forecastCharts.arima.data.datasets[2].data = employment.forecast_mean;
            }
            
            this.forecastCharts.arima.update('none');
        }
        
        // Update GBM Chart (full timeline)
        if (this.data.forecasts.gbm) {
            this.forecastCharts.gbm.data.labels = allYears;
            
            // Nominal Value Added
            const nominalValueAdded = this.data.forecasts.gbm.nominal_value_added;
            if (nominalValueAdded && nominalValueAdded.forecast_mean) {
                this.forecastCharts.gbm.data.datasets[0].data = nominalValueAdded.forecast_mean;
            }
            
            // Nominal Gross Output
            const nominalGrossOutput = this.data.forecasts.gbm.nominal_gross_output;
            if (nominalGrossOutput && nominalGrossOutput.forecast_mean) {
                this.forecastCharts.gbm.data.datasets[1].data = nominalGrossOutput.forecast_mean;
            }
            
            this.forecastCharts.gbm.update('none');
        }
    }
    
    updateStats() {
        const yearIndex = this.data.years.indexOf(this.currentYear);
        if (yearIndex === -1) return;
        
        const economicData = this.data.data.economic_output;
        const laborData = this.data.data.labor_market;
        
        const currentValueAdded = economicData.real_value_added[yearIndex] || 0;
        const currentEmployment = laborData.employment[yearIndex] || 0;
        const currentCompensation = laborData.compensation[yearIndex] || 0;
        
        const productivity = currentEmployment > 0 ? (currentValueAdded / currentEmployment).toFixed(1) : 0;
        
        const statsHtml = `
            <div class="stat-item">
                <div class="stat-value">$${(currentValueAdded / 1000).toFixed(1)}B</div>
                <div class="stat-label">Real Value Added</div>
            </div>
            <div class="stat-item">
                <div class="stat-value">${currentEmployment}K</div>
                <div class="stat-label">Employment</div>
            </div>
            <div class="stat-item">
                <div class="stat-value">$${currentCompensation}M</div>
                <div class="stat-label">Compensation</div>
            </div>
            <div class="stat-item">
                <div class="stat-value">$${productivity}K</div>
                <div class="stat-label">Productivity per Worker</div>
            </div>
        `;
        
        document.getElementById('statsDisplay').innerHTML = statsHtml;
    }
    
    startAnimation() {
        if (this.isPlaying) return;
        
        this.isPlaying = true;
        const playBtn = document.getElementById('playBtn');
        playBtn.textContent = '⏸️ Pause';
        
        const startYear = Math.min(...this.data.years);
        const endYear = Math.max(...this.data.years);
        
        // Start from beginning if at the end
        if (this.currentYear >= endYear) {
            this.currentYear = startYear;
        }
        
        this.playInterval = setInterval(() => {
            if (this.currentYear >= endYear) {
                this.stopAnimation();
                return;
            }
            
            this.currentYear++;
            document.getElementById('yearSlider').value = this.currentYear;
            document.getElementById('yearDisplay').textContent = this.currentYear;
            this.updateAllCharts();
            this.updateStats();
            this.updateYearDisplay();
        }, this.animationSpeed);
    }
    
    stopAnimation() {
        this.isPlaying = false;
        const playBtn = document.getElementById('playBtn');
        playBtn.textContent = '▶️ Play Animation';
        
        if (this.playInterval) {
            clearInterval(this.playInterval);
            this.playInterval = null;
        }
    }
    
    resetToStart() {
        this.stopAnimation();
        this.currentYear = Math.min(...this.data.years);
        document.getElementById('yearSlider').value = this.currentYear;
        document.getElementById('yearDisplay').textContent = this.currentYear;
        this.updateAllCharts();
        this.updateStats();
        this.updateYearDisplay();
    }
}

// Initialize the application when the page loads
document.addEventListener('DOMContentLoaded', () => {
    new SpaceEconomyApp();
});