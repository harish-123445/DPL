# Data Premier League

# F1 Data Explorer

## Overview

F1 Data Explorer is a Streamlit-based interactive dashboard for analyzing historical Formula 1 World Championship data. It provides insights into driver and constructor performance, qualifying vs. race results, pit stop strategies, and hypothetical driver swaps.

## Features

- **Dataset Overview:** Load and analyze various datasets related to Formula 1.
- **Feature Engineering Insights:**
  - Driver & Constructor Performance Analysis
  - Qualifying vs. Race Performance Trends
  - Pit Stop Strategy Optimization
  - Head-to-Head Driver Analysis
  - Hypothetical Driver Swaps & Performance Prediction
  - Driver Movements & Team Networks
  - Team Performance Comparisons
  - Lap Time Efficiency Analysis
  - Championship Retention Probability
  - Future Season Predictions
- **Interactive Visualizations:**
  - Correlation heatmaps
  - Scatter plots
  - Bar charts
  - Network graphs

## Datasets Used

The following CSV files are used for data analysis:

- `circuits.csv`
- `constructor_results.csv`
- `constructor_standings.csv`
- `constructors.csv`
- `driver_standings.csv`
- `drivers.csv`
- `lap_times.csv`
- `pit_stops.csv`
- `qualifying.csv`
- `races.csv`
- `results.csv`
- `seasons.csv`
- `sprint_results.csv`
- `status.csv`

## Installation & Usage

### Prerequisites

- Python 3.7+
- Streamlit
- Pandas
- Seaborn
- Matplotlib

### Setup

1. Clone the repository:
   ```sh
   git clone https://github.com/your-repo/F1-Data-Explorer.git
   cd F1-Data-Explorer
   ```
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
3. Run the Streamlit application:
   ```sh
   streamlit run main.py
   ```

### Usage

1. **Dataset Selection:**
   - Select a dataset from the sidebar to explore detailed insights.
   - View statistical summaries, missing values, and correlation heatmaps.

2. **Feature Engineering Analysis:**
   - Navigate through different analysis sections to understand driver and constructor performance.
   - Compare lap time efficiencies and driver consistency.
   
3. **Interactive Visualizations:**
   - Use filters and selectors to customize visualizations.
   - Analyze team and driver performances dynamically.

4. **Hypothetical Driver Swaps:**
   - Simulate driver swaps and predict their potential performance changes.

5. **Predictions for Future Seasons:**
   - Forecast upcoming season outcomes using historical data trends.

## Customization

To modify the dashboard, update `main.py` to adjust dataset file paths, feature engineering methods, or visualization styles.




