import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# Set page configuration
st.set_page_config(
    page_title="F1 Data Explorer",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Define dataset files dictionary
dataset_files = {
    "Circuits": "DPL_Datasets/circuits.csv",
    "Constructor Results": "DPL_Datasets/constructor_results.csv",
    "Constructor Standings": "DPL_Datasets/constructor_standings.csv",
    "Constructors": "DPL_Datasets/constructors.csv",
    "Driver Standings": "DPL_Datasets/driver_standings.csv",
    "Drivers": "DPL_Datasets/drivers.csv",
    "Lap Times": "DPL_Datasets/lap_times.csv",
    "Pit Stops": "DPL_Datasets/pit_stops.csv",
    "Qualifying": "DPL_Datasets/qualifying.csv",
    "Races": "DPL_Datasets/races.csv",
    "Results": "DPL_Datasets/results.csv",
    "Seasons": "DPL_Datasets/seasons.csv",
    "Sprint Results": "DPL_Datasets/sprint_results.csv",
    "Status": "DPL_Datasets/status.csv",
}

# Feature Engineering Sections
feature_sections = {
    "Driver & Constructor Performance": "Analyze win ratios, podium finishes, and career longevity.",
    "Qualifying vs. Race Performance": "Impact of grid position on race results and position gains.",
    "Pit Stop Strategies": "Optimal pit stop frequency and efficiency analysis.",
    "Head-to-Head Driver Analysis": "Competitive rivalries and head-to-head race stats.",
    "Hypothetical Driver Swaps": "Swapping drivers and predicting performance impact.",
    "Driver Movements & Team Networks":"Map driver transitions across teams use network graph for visualizations.",
    "Team Performance Comparison": "Compare team success rates against different opponents with and without considering circuit factor.",
    "Driver Consistency in Race Performance": "Identify drivers with consistent top finishes and those with fluctuating results.",
    "Lap Time Efficiency Analysis": "Compare lap times across different circuits and identify which teams maximize efficiency.",
    "Best Team Lineup":"Build the best possible team lineup based on driver performance trends.",
    "Predictions for 2025 Season": "Predict the top  drivers and constructors for the upcoming season.",
    "Struggling Teams Analysis": "Analyze the performance of struggling teams and identify areas for improvement.",
    "Driver-Specific Track Struggles": "Identify tracks where specific drivers struggle and excel.",
    "Championship Retention Probability": "Predict the probability of a driver or constructor retaining their championship title.",
    "Champion Age Trends": "Analyze the age trends of F1 champions and predict future champions.",
    "Bonus Challenge (Optional)": "Predict the future team of a driver based on past team transitions and transfer trends."
}

# Function to load data
def load_data(file_path):
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

# Custom CSS for enhanced styling
def add_custom_css():
    st.markdown("""
    <style>
        /* General styling */
        body {
            background-color: #0E1117;
            color: #FFFFFF;
            font-family: 'Helvetica Neue', Arial, sans-serif;
        }
        
        /* Header styling */
        .main-header {
            background: linear-gradient(90deg, #E10600 0%, #FF8700 100%);
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 30px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
            text-align: center;
        }
        
        .main-title {
            font-size: 3.5rem;
            font-weight: 800;
            color: white;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
        }
        
        .sub-title {
            font-size: 1.5rem;
            color: #f0f0f0;
            font-weight: 300;
            margin-bottom: 15px;
        }
        
        /* Card styling */
        .card-container {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-top: 30px;
        }
        
        .card {
            background: rgba(255, 255, 255, 0.1);
            border-radius: 15px;
            padding: 25px;
            transition: transform 0.3s, box-shadow 0.3s;
            height: 100%;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        
        .card:hover {
            transform: translateY(-5px);
            box-shadow: 0 10px 20px rgba(0, 0, 0, 0.2);
            border: 1px solid rgba(255, 255, 255, 0.2);
        }
        
        .card-title {
            font-size: 1.8rem;
            font-weight: 600;
            margin-bottom: 15px;
            color: #E10600;
        }
        
        .card-content {
            font-size: 1rem;
            color: #cccccc;
            margin-bottom: 20px;
            flex-grow: 1;
        }
        
        .card-button {
            background: linear-gradient(90deg, #E10600 0%, #FF8700 100%);
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 5px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s;
            display: block;
            text-align: center;
            margin-top: auto;
            text-decoration: none;
        }
        
        .card-button:hover {
            opacity: 0.9;
            transform: scale(1.05);
        }
        
        /* Dashboard elements styling */
        .metric-container {
            background: rgba(255, 255, 255, 0.07);
            border-radius: 10px;
            padding: 15px;
            margin-bottom: 20px;
        }
        
        .metric-title {
            font-weight: 600;
            color: #E10600;
            margin-bottom: 5px;
        }
        
        .metric-value {
            font-size: 2rem;
            font-weight: 700;
        }
        
        .section-header {
            border-left: 5px solid #E10600;
            padding-left: 15px;
            margin: 30px 0 20px 0;
            font-size: 1.8rem;
            font-weight: 600;
        }
        
        /* Navigation breadcrumb */
        .breadcrumb {
            display: flex;
            align-items: center;
            margin-bottom: 20px;
            font-size: 0.9rem;
        }
        
        .breadcrumb a {
            color: #CCCCCC;
            text-decoration: none;
            transition: color 0.3s;
        }
        
        .breadcrumb a:hover {
            color: #E10600;
        }
        
        .breadcrumb span {
            margin: 0 10px;
            color: #666666;
        }
        
        /* Animation */
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .animate-fadeIn {
            animation: fadeIn 0.6s ease-out forwards;
        }
        
        /* Loading spinner */
        .loading-spinner {
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100px;
        }
        
        /* Tab styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
        }

        .stTabs [data-baseweb="tab"] {
            background-color: rgba(255, 255, 255, 0.05);
            border-radius: 4px 4px 0 0;
            padding: 10px 16px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-bottom: none;
        }

        .stTabs [aria-selected="true"] {
            background-color: rgba(225, 6, 0, 0.1);
            border-bottom: 2px solid #E10600;
        }
    </style>
    """, unsafe_allow_html=True)

# Create animated background
def add_animated_background():
    st.markdown("""
    <div class="background">
        <span></span>
        <span></span>
        <span></span>
        <span></span>
        <span></span>
        <span></span>
    </div>
    
    <style>
    .background {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        z-index: -1;
        overflow: hidden;
    }
    
    .background span {
        position: absolute;
        width: 20vmin;
        height: 20vmin;
        border-radius: 20vmin;
        backface-visibility: hidden;
        opacity: 0.05;
        animation: move 8s infinite;
        background: #E10600;
    }
    
    .background span:nth-child(1) {
        top: 10%;
        left: 10%;
        animation-duration: 12s;
        animation-delay: 0s;
    }
    
    .background span:nth-child(2) {
        top: 20%;
        left: 80%;
        animation-duration: 15s;
        animation-delay: 0.3s;
        background: #FF8700;
    }
    
    .background span:nth-child(3) {
        top: 70%;
        left: 20%;
        animation-duration: 10s;
        animation-delay: 0.7s;
        background: #1E5BC6;
    }
    
    .background span:nth-child(4) {
        top: 60%;
        left: 70%;
        animation-duration: 18s;
        animation-delay: 0.1s;
        background: #E10600;
    }
    
    .background span:nth-child(5) {
        top: 40%;
        left: 40%;
        animation-duration: 14s;
        animation-delay: 0.5s;
        background: #FF8700;
    }
    
    .background span:nth-child(6) {
        top: 90%;
        left: 10%;
        animation-duration: 12s;
        animation-delay: 0.8s;
        background: #1E5BC6;
    }
    
    @keyframes move {
        0% {
            transform: translate(0, 0) scale(1);
        }
        25% {
            transform: translate(10vw, -5vh) scale(1.1);
        }
        50% {
            transform: translate(5vw, 10vh) scale(0.9);
        }
        75% {
            transform: translate(-10vw, 5vh) scale(1.2);
        }
        100% {
            transform: translate(0, 0) scale(1);
        }
    }
    </style>
    """, unsafe_allow_html=True)

# Create breadcrumb navigation
def add_breadcrumb(items=None):
    if items is None:
        items = [("Home", "home")]
    
    breadcrumb_html = '<div class="breadcrumb">'
    for i, (name, url) in enumerate(items):
        if i > 0:
            breadcrumb_html += '<span>/</span>'
        
        # Make the last item non-clickable
        if i == len(items) - 1:
            breadcrumb_html += f'<span>{name}</span>'
        else:
            breadcrumb_html += f'<a href="#{url}">{name}</a>'
    
    breadcrumb_html += '</div>'
    st.markdown(breadcrumb_html, unsafe_allow_html=True)

# Function to create a card
def create_card(title, description, button_text, key):
    return f"""
    <div class="card animate-fadeIn">
        <h3 class="card-title">{title}</h3>
        <p class="card-content">{description}</p>
    </div>
    """

# Create a section header
def section_header(title):
    st.markdown(f'<h2 class="section-header">{title}</h2>', unsafe_allow_html=True)

# Dashboard for datasets
def dataset_dashboard(dataset_name):
    add_breadcrumb([("Home", "home"), ("Datasets", "datasets"), (dataset_name, f"dataset_{dataset_name}")])
    
    # Header
    st.markdown(f'<h1 class="main-title">{dataset_name} Dataset</h1>', unsafe_allow_html=True)
    
    # Loading spinner
    with st.spinner(f"Loading {dataset_name} dataset..."):
        df = load_data(dataset_files[dataset_name])
    
    # Quick stats in a grid
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(
            f"""
            <div class="metric-container">
                <div class="metric-title">Rows</div>
                <div class="metric-value">{df.shape[0]:,}</div>
            </div>
            """, 
            unsafe_allow_html=True
        )
    
    with col2:
        st.markdown(
            f"""
            <div class="metric-container">
                <div class="metric-title">Columns</div>
                <div class="metric-value">{df.shape[1]}</div>
            </div>
            """, 
            unsafe_allow_html=True
        )
    
    with col3:
        missing = df.isnull().sum().sum()
        st.markdown(
            f"""
            <div class="metric-container">
                <div class="metric-title">Missing Values</div>
                <div class="metric-value">{missing:,}</div>
            </div>
            """, 
            unsafe_allow_html=True
        )
    
    with col4:
        memory = df.memory_usage(deep=True).sum()
        st.markdown(
            f"""
            <div class="metric-container">
                <div class="metric-title">Memory Usage</div>
                <div class="metric-value">{memory / 1024**2:.2f} MB</div>
            </div>
            """, 
            unsafe_allow_html=True
        )
    
    # Create tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Preview", "📈 Statistics", "🔍 Missing Values", "🔄 Correlations"])
    
    with tab1:
        st.dataframe(df.head(10), use_container_width=True)
        
        # Column information
        section_header("Column Information")
        col_info = pd.DataFrame({
            'Column': df.columns,
            'Type': df.dtypes,
            'Non-Null Count': df.count(),
            'Null Count': df.isnull().sum(),
            'Unique Values': [df[col].nunique() for col in df.columns]
        })
        st.dataframe(col_info, use_container_width=True)
    
    with tab2:
        # Summary statistics
        section_header("Summary Statistics")
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        if len(numeric_cols) > 0:
            st.dataframe(df[numeric_cols].describe(), use_container_width=True)
        else:
            st.info("No numeric columns found in this dataset.")
            
        # Categorical columns stats
        cat_cols = df.select_dtypes(include=['object', 'category']).columns
        if len(cat_cols) > 0:
            section_header("Categorical Columns")
            selected_cat_col = st.selectbox("Select a categorical column:", cat_cols)
            value_counts = df[selected_cat_col].value_counts().reset_index()
            value_counts.columns = [selected_cat_col, 'Count']
            
            # Show value counts and visualization
            col1, col2 = st.columns([1, 2])
            with col1:
                st.dataframe(value_counts.head(20), use_container_width=True)
            with col2:
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.barplot(x=value_counts[selected_cat_col].head(15), y=value_counts['Count'].head(15), ax=ax)
                plt.xticks(rotation=45, ha='right')
                plt.title(f'Top 15 values for {selected_cat_col}')
                plt.tight_layout()
                st.pyplot(fig)
    
    with tab3:
        # Missing values visualization
        section_header("Missing Values")
        
        col1, col2 = st.columns([1, 2])
        with col1:
            missing_df = pd.DataFrame({
                'Column': df.columns,
                'Missing Values': df.isnull().sum(),
                'Percentage': df.isnull().sum() / len(df) * 100
            }).sort_values('Missing Values', ascending=False)
            
            st.dataframe(missing_df, use_container_width=True)
        
        with col2:
            if missing_df['Missing Values'].sum() > 0:
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.heatmap(df.isnull(), cmap='viridis', cbar=False, ax=ax)
                plt.title('Missing Values Heatmap')
                plt.tight_layout()
                st.pyplot(fig)
            else:
                st.success("No missing values in this dataset!")
    
    with tab4:
        # Correlation analysis for numeric columns
        section_header("Correlation Analysis")
        
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        if len(numeric_cols) > 1:
            corr_matrix = df[numeric_cols].corr()
            
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5, ax=ax)
            plt.title('Correlation Matrix')
            plt.tight_layout()
            st.pyplot(fig)
            
            st.write("### Strongest Correlations")
            # Get the strongest correlations (excluding self-correlations)
            corr_unstack = corr_matrix.unstack()
            corr_unstack = corr_unstack[corr_unstack < 1.0]  # Remove self-correlations
            strongest_corrs = corr_unstack.abs().sort_values(ascending=False)[:15]
            
            if not strongest_corrs.empty:
                for (col1, col2), corr_value in strongest_corrs.items():
                    st.write(f"**{col1}** and **{col2}**: {corr_value:.3f}")
                    
                # Scatter plot of the strongest correlation
                strongest_pair = strongest_corrs.index[0]
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.scatterplot(x=df[strongest_pair[0]], y=df[strongest_pair[1]], ax=ax)
                plt.title(f'Scatter Plot: {strongest_pair[0]} vs {strongest_pair[1]}')
                plt.tight_layout()
                st.pyplot(fig)
            else:
                st.info("No significant correlations found.")
        else:
            st.info("At least two numeric columns are required for correlation analysis.")
    
    st.markdown('</div>', unsafe_allow_html=True)
    col1, col2,col3 = st.columns(3)
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        with st.container():
            co1, co2, co3 = st.columns(3)
            with co2:
                if st.button("Back to Home", key="back_to_home", help="Return to home page"):
                    st.session_state.page = "home"
                    st.rerun()

# Feature engineering dashboard
def feature_dashboard(feature_name):
    add_breadcrumb([("Home", "home"), ("Feature Engineering", "features"), (feature_name, f"feature_{feature_name}")])
    

    # Placeholder visualizations
    if feature_name == "Driver & Constructor Performance":
        import pandas as pd
        import matplotlib.pyplot as plt
        import seaborn as sns

        # Load data
        results = load_data('DPL_Datasets/results.csv')
        races = load_data('DPL_Datasets/races.csv')
        drivers = load_data('DPL_Datasets/drivers.csv')
        constructors = load_data('DPL_Datasets/constructors.csv')
        # Merge race year
        results = results.merge(races[['raceId', 'year']], on='raceId', how='left')

        # Driver Performance Metrics
        driver_wins = results[results['positionOrder'] == 1].groupby('driverId').size()
        total_races = results.groupby('driverId').size()
        win_ratio = (driver_wins / total_races).fillna(0)

        podiums = results[results['positionOrder'] <= 3].groupby('driverId').size()
        career_span = results.groupby('driverId')['year'].agg(['min', 'max'])
        career_span['years_active'] = career_span['max'] - career_span['min'] + 1
        total_points = results.groupby('driverId')['points'].sum()

        driver_performance = pd.DataFrame({
            'win_ratio': win_ratio,
            'podiums': podiums,
            'career_years': career_span['years_active'],
            'total_points': total_points
        }).fillna(0)

        driver_performance = driver_performance.merge(drivers[['driverId', 'forename', 'surname']], on='driverId')
        driver_performance['Driver Name'] = driver_performance['forename'] + " " + driver_performance['surname']
        driver_performance.drop(columns=['forename', 'surname'], inplace=True)

        # Constructor Performance Metrics
        constructor_wins = results[results['positionOrder'] == 1].groupby('constructorId').size()
        total_races_constructor = results.groupby('constructorId').size()
        constructor_win_ratio = (constructor_wins / total_races_constructor).fillna(0)
        constructor_podiums = results[results['positionOrder'] <= 3].groupby('constructorId').size()
        constructor_total_points = results.groupby('constructorId')['points'].sum()

        constructor_performance = pd.DataFrame({
            'win_ratio': constructor_win_ratio,
            'podiums': constructor_podiums,
            'total_points': constructor_total_points
        }).fillna(0)

        constructor_performance = constructor_performance.merge(constructors[['constructorId', 'name']], on='constructorId')

        # Streamlit App
        st.title("🏎️ F1 Driver & Constructor Performance Analysis")

        # Top Drivers by Win Ratio
        st.subheader("Top 10 Drivers by Win Ratio")
        st.dataframe(driver_performance.sort_values(by='win_ratio', ascending=False).head(10))

        # Top Constructors by Win Ratio
        st.subheader("Top 10 Constructors by Win Ratio")
        st.dataframe(constructor_performance.sort_values(by='win_ratio', ascending=False).head(10))

        # Correlation Heatmap
        st.subheader("📊 Correlation Between Career Longevity & Success Metrics")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            driver_performance[['career_years', 'win_ratio', 'podiums', 'total_points']].corr(),
            annot=True, cmap='coolwarm', linewidths=0.5, ax=ax
        )
        st.pyplot(fig)

    
    elif feature_name == "Qualifying vs. Race Performance":
        #import streamlit as st
        import pandas as pd
        import matplotlib.pyplot as plt
        import seaborn as sns

        # Load Data
        results = load_data('DPL_Datasets/results.csv')
        races = load_data('DPL_Datasets/races.csv')
        drivers = load_data('DPL_Datasets/drivers.csv')
        constructors = load_data('DPL_Datasets/constructors.csv')
        qualifying = load_data('DPL_Datasets/qualifying.csv')
        # Merge race year
        results = results.merge(races[['raceId', 'year']], on='raceId', how='left')

        # Driver Performance Metrics
        driver_wins = results[results['positionOrder'] == 1].groupby('driverId').size()
        total_races = results.groupby('driverId').size()
        win_ratio = (driver_wins / total_races).fillna(0)

        podiums = results[results['positionOrder'] <= 3].groupby('driverId').size()
        career_span = results.groupby('driverId')['year'].agg(['min', 'max'])
        career_span['years_active'] = career_span['max'] - career_span['min'] + 1
        total_points = results.groupby('driverId')['points'].sum()

        driver_performance = pd.DataFrame({
            'win_ratio': win_ratio,
            'podiums': podiums,
            'career_years': career_span['years_active'],
            'total_points': total_points
        }).fillna(0)

        driver_performance = driver_performance.merge(drivers[['driverId', 'forename', 'surname']], on='driverId')
        driver_performance['Driver Name'] = driver_performance['forename'] + " " + driver_performance['surname']
        driver_performance.drop(columns=['forename', 'surname'], inplace=True)

        # Constructor Performance Metrics
        constructor_wins = results[results['positionOrder'] == 1].groupby('constructorId').size()
        total_races_constructor = results.groupby('constructorId').size()
        constructor_win_ratio = (constructor_wins / total_races_constructor).fillna(0)
        constructor_podiums = results[results['positionOrder'] <= 3].groupby('constructorId').size()
        constructor_total_points = results.groupby('constructorId')['points'].sum()

        constructor_performance = pd.DataFrame({
            'win_ratio': constructor_win_ratio,
            'podiums': constructor_podiums,
            'total_points': constructor_total_points
        }).fillna(0)

        constructor_performance = constructor_performance.merge(constructors[['constructorId', 'name']], on='constructorId')

        # Streamlit App
        st.title("Qualifying vs. Race Performance Analysis")


        # Trend Analysis
        st.subheader("📈 Impact of Grid Position on Final Race Position")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.regplot(x=results['grid'], y=results['positionOrder'], scatter_kws={'alpha':0.5}, line_kws={"color": "red"}, ax=ax)
        ax.set_title("Impact of Grid Position on Final Race Position", fontsize=14)
        ax.set_xlabel("Starting Grid Position (Lower is Better)", fontsize=12)
        ax.set_ylabel("Final Race Position (Lower is Better)", fontsize=12)
        ax.invert_xaxis()
        ax.invert_yaxis()
        st.pyplot(fig)

        # Positions Gained Analysis
        qualifying_results = qualifying[['raceId', 'driverId', 'position']].rename(columns={'position': 'grid_position'})
        race_results = results[['raceId', 'driverId', 'positionOrder']].rename(columns={'positionOrder': 'final_position'})
        merged_data = pd.merge(qualifying_results, race_results, on=['raceId', 'driverId'])
        merged_data['positions_gained'] = merged_data['grid_position'] - merged_data['final_position']
        driver_performance = merged_data.groupby('driverId')['positions_gained'].mean().sort_values(ascending=False)

        st.subheader("🔝 Top 10 Drivers Who Gained Most Positions")
        st.dataframe(driver_performance.head(10))

        # Correlation Between Grid Position and Final Position
        correlation = merged_data[['grid_position', 'final_position']].corr().iloc[0,1]
        st.subheader("📊 Correlation Between Grid & Final Position")
        st.write(f"Correlation: {correlation:.2f}")

        # Heatmap of Grid Position vs Final Position
        st.subheader("🔥 Heatmap of Grid Position vs Final Race Position")
        heatmap_data = merged_data.pivot_table(index='final_position', columns='grid_position', aggfunc='size', fill_value=0)
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(heatmap_data, cmap="YlGnBu", linewidths=0.5, annot=False, ax=ax)
        ax.set_xlabel("Starting Grid Position (Lower is Better)")
        ax.set_ylabel("Final Race Position (Lower is Better)")
        ax.set_title("Heatmap of Grid Position vs Final Race Position")
        st.pyplot(fig)
    
    elif feature_name == "Pit Stop Strategies":
        import pandas as pd
        import matplotlib.pyplot as plt
        import seaborn as sns

        # Load Data
        pit_stops = load_data('DPL_Datasets/pit_stops.csv')
        results = load_data('DPL_Datasets/results.csv')

        st.title("🏎️ F1 Pit Stop Analysis")

        # Pit Stop Frequency per Race
        pit_stop_counts = pit_stops.groupby(['raceId', 'driverId'])['stop'].max().reset_index()
        avg_pit_stops = pit_stop_counts['stop'].mean()
        st.subheader("📊 Average Pit Stops per Driver per Race")
        st.write(f"Average Pit Stops per Driver per Race: {avg_pit_stops:.2f}")

        # Distribution of Pit Stop Lap Timing
        st.subheader("📈 Distribution of Pit Stop Lap Timing")
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.histplot(pit_stops['lap'], bins=30, kde=True, color='blue', ax=ax)
        ax.set_title('Distribution of Pit Stop Lap Timing')
        ax.set_xlabel('Lap Number')
        ax.set_ylabel('Frequency')
        st.pyplot(fig)

        # Pit Stop Duration Analysis
        st.subheader("⏳ Pit Stop Duration by Stop Number")
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.boxplot(x=pit_stops['stop'], y=pit_stops['milliseconds'] / 1000, palette='coolwarm', ax=ax)
        ax.set_title('Pit Stop Duration by Stop Number')
        ax.set_xlabel('Pit Stop Number')
        ax.set_ylabel('Duration (seconds)')
        st.pyplot(fig)

        # Impact of Pit Stop Duration on Final Race Position
        st.subheader("🚦 Impact of Pit Stop Time on Final Race Position")
        merged_data = pit_stops.merge(results, on=['raceId', 'driverId'])
        merged_data['pit_stop_time'] = merged_data.groupby(['raceId', 'driverId'])['milliseconds_x'].transform('sum')

        fig, ax = plt.subplots(figsize=(12, 6))
        sns.scatterplot(x=merged_data['pit_stop_time'] / 1000, y=merged_data['positionOrder'], alpha=0.6, ax=ax)
        ax.set_title('Impact of Pit Stop Time on Final Race Position')
        ax.set_xlabel('Total Pit Stop Time (seconds)')
        ax.set_ylabel('Final Race Position (Lower is Better)')
        st.pyplot(fig)
    elif feature_name == "Head-to-Head Driver Analysis":
        #import streamlit as st
        import pandas as pd
        import matplotlib.pyplot as plt
        import seaborn as sns
        from itertools import combinations

        results = load_data('DPL_Datasets/results.csv')

        st.title("🏎️ Head-to-Head Driver Analysis")

        # Extract relevant race results
        race_results = results[['raceId', 'driverId', 'positionOrder']]

        # Function to generate driver pairs per race
        def get_driver_pairs(group):
            drivers = group[['driverId', 'positionOrder']].values
            pairs = list(combinations(drivers, 2))
            return pd.DataFrame([(group.name, d1[0], d2[0], d1[1], d2[1]) for d1, d2 in pairs],
                                columns=['raceId', 'driver1', 'driver2', 'pos1', 'pos2'])

        # Generate driver comparisons
        driver_comparisons = race_results.groupby('raceId').apply(get_driver_pairs).reset_index(drop=True)

        # Determine head-to-head winners
        driver_comparisons['winner'] = driver_comparisons.apply(lambda x: x['driver1'] if x['pos1'] < x['pos2'] else x['driver2'], axis=1)

        # Count head-to-head wins
        head_to_head = driver_comparisons.groupby(['driver1', 'driver2'])['winner'].value_counts().unstack(fill_value=0)

        # Calculate win ratio
        head_to_head['total_races'] = head_to_head.sum(axis=1)
        head_to_head['win_ratio_driver1'] = head_to_head.iloc[:, 0] / head_to_head['total_races']
        head_to_head['win_ratio_driver2'] = head_to_head.iloc[:, 1] / head_to_head['total_races']

        # Compute competitiveness score
        head_to_head['competitiveness'] = abs(head_to_head['win_ratio_driver1'] - head_to_head['win_ratio_driver2'])
        most_competitive = head_to_head.sort_values('competitiveness', ascending=True)

        # Display top 10 rivalries
        st.subheader("🔥 Top 10 Most Competitive Driver Rivalries")
        st.dataframe(most_competitive.head(10))

        # Visualization: Heatmap of Rivalry Wins
        st.subheader("📊 Head-to-Head Win Counts Heatmap")
        plt.figure(figsize=(10, 8))
        sns.heatmap(head_to_head.iloc[:, :2], cmap="coolwarm", annot=True, fmt="d")
        plt.xlabel("Drivers")
        plt.ylabel("Rival Drivers")
        plt.title("Head-to-Head Win Counts")
        st.pyplot(plt)
    elif feature_name == "Hypothetical Driver Swaps":
        #import streamlit as st
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.metrics import r2_score
        from sklearn.preprocessing import LabelEncoder

        # Load Data
        results = load_data('DPL_Datasets/results.csv')
        qualifying = load_data('DPL_Datasets/qualifying.csv')
        

        st.title("🏎️ Driver Swap Analysis & Performance Prediction")

        # Merge datasets
        data = results[['raceId', 'driverId', 'constructorId', 'grid', 'fastestLapTime', 'points']]
        data = data.merge(qualifying[['raceId', 'driverId', 'position']], on=['raceId', 'driverId'], how='left')
        data.rename(columns={'position': 'qualifying_position'}, inplace=True)

        # Convert fastestLapTime to numerical values
        def time_to_ms(time_str):
            try:
                m, s = time_str.split(':')
                return int(m) * 60000 + float(s) * 1000
            except:
                return np.nan  # Handle NaN values

        data['fastestLapTime'] = data['fastestLapTime'].apply(time_to_ms)

        # Fill missing values with median
        data.fillna(data.median(), inplace=True)

        # Encode categorical variables
        encoder = LabelEncoder()
        data['driverId'] = encoder.fit_transform(data['driverId'])
        data['constructorId'] = encoder.fit_transform(data['constructorId'])

        # Define features and target variable
        X = data[['driverId', 'constructorId', 'grid', 'qualifying_position', 'fastestLapTime']]
        y = data['points']

        # Split dataset
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Predictions & Performance
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)

        st.subheader("📊 Model Performance")
        st.write(f"**R² Score:** {r2:.3f}")

        # --- DRIVER SWAP ANALYSIS ---
        st.subheader("🔄 Driver Swap Analysis")

        # Select two drivers
        driver_options = data['driverId'].unique()
        driver_1 = st.selectbox("Select Driver 1:", driver_options)
        driver_2 = st.selectbox("Select Driver 2:", driver_options)

        if driver_1 != driver_2:
            # Get their current teams
            team_1 = results.loc[results['driverId'] == driver_1, 'constructorId'].mode()[0]
            team_2 = results.loc[results['driverId'] == driver_2, 'constructorId'].mode()[0]

            # Create swapped dataset
            swapped_data = data.copy()
            swapped_data.loc[swapped_data['driverId'] == driver_1, 'constructorId'] = team_2
            swapped_data.loc[swapped_data['driverId'] == driver_2, 'constructorId'] = team_1

            # Predict new points after swap
            swapped_data['predicted_points'] = model.predict(swapped_data[['driverId', 'constructorId', 'grid', 'qualifying_position', 'fastestLapTime']])

            # Compare original vs swapped driver points
            original_points = data.groupby('driverId')['points'].sum()
            new_points = swapped_data.groupby('driverId')['predicted_points'].sum()

            comparison = pd.DataFrame({'original_points': original_points, 'new_points': new_points})
            comparison['point_change'] = comparison['new_points'] - comparison['original_points']

            st.write(f"**Impact of Driver Swap (Driver {driver_1} & Driver {driver_2}):**")
            st.dataframe(comparison.loc[[driver_1, driver_2]])

            # --- Heatmap of Grid Position vs Final Points ---
            st.subheader("🔥 Impact of Grid Position on Points")
            plt.figure(figsize=(10, 6))
            sns.heatmap(data.pivot_table(index='grid', values='points', aggfunc='mean'), cmap="coolwarm", annot=True)
            plt.title("Grid Position vs Final Points")
            st.pyplot(plt)

            # --- Histogram of Points Change ---
            st.subheader("📈 Distribution of Points Change Due to Driver Swap")
            plt.figure(figsize=(10, 6))
            sns.histplot(comparison['point_change'], bins=20, kde=True, color="blue")
            plt.xlabel("Points Change")
            plt.ylabel("Frequency")
            plt.title("Distribution of Points Change After Driver Swap")
            st.pyplot(plt)

            # --- Bar Chart: Points Before vs. After Swap ---
            st.subheader("📊 Driver Points Before vs. After Swap")
            swapped_drivers = comparison.loc[[driver_1, driver_2]]

            plt.figure(figsize=(8, 5))
            swapped_drivers[['original_points', 'new_points']].plot(kind='bar', figsize=(8, 5), color=['blue', 'green'])
            plt.xlabel("Driver ID")
            plt.ylabel("Total Points")
            plt.xticks(rotation=0)
            plt.legend(["Original Points", "Predicted Points After Swap"])
            plt.title("Driver Performance Comparison After Swap")
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            st.pyplot(plt)
    elif feature_name=="Driver Movements & Team Networks":
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        import seaborn as sns
        import networkx as nx

        # Load datasets
        results = load_data('DPL_Datasets/results.csv')
        driver_standings = load_data('DPL_Datasets/driver_standings.csv')
        constructor_standings = load_data('DPL_Datasets/constructor_standings.csv')
        

        st.title("🔄 Driver Swap & Team Transitions")

        # Select two drivers
        driver_options = results['driverId'].unique()
        driver1_id = st.selectbox("Select Driver 1:", driver_options)
        driver2_id = st.selectbox("Select Driver 2:", driver_options)

        # Function to swap drivers and analyze impact
        def swap_drivers_and_analyze(driver1_id, driver2_id, results, driver_standings, constructor_standings):
            temp_team = results.loc[results['driverId'] == driver1_id, 'constructorId'].values[0]
            results.loc[results['driverId'] == driver1_id, 'constructorId'] = results.loc[results['driverId'] == driver2_id, 'constructorId'].values[0]
            results.loc[results['driverId'] == driver2_id, 'constructorId'] = temp_team

            # Recalculate driver standings
            new_driver_standings = results.groupby('driverId')['points'].sum().reset_index()
            driver_standings = driver_standings.drop(columns=['points']).merge(new_driver_standings, on='driverId', how='left')

            # Recalculate constructor standings
            new_constructor_standings = results.groupby('constructorId')['points'].sum().reset_index()
            constructor_standings = constructor_standings.drop(columns=['points']).merge(new_constructor_standings, on='constructorId', how='left')

            return driver_standings, constructor_standings

        if st.button("🔄 Swap Drivers & Analyze"):
            driver_standings_updated, constructor_standings_updated = swap_drivers_and_analyze(driver1_id, driver2_id, results, driver_standings, constructor_standings)

            st.subheader("📊 Updated Driver Standings")
            st.dataframe(driver_standings_updated.sort_values(by="points", ascending=False))

            st.subheader("🏎️ Updated Constructor Standings")
            st.dataframe(constructor_standings_updated.sort_values(by="points", ascending=False))

            # ---- Network Graph for Driver Transitions ----
            st.subheader("🔗 Driver Transitions Across Teams")
            G = nx.DiGraph()

            for driver_id in results['driverId'].unique():
                teams = results[results['driverId'] == driver_id]['constructorId'].unique()
                for i in range(len(teams) - 1):
                    G.add_edge(teams[i], teams[i + 1], driver=driver_id)

            plt.figure(figsize=(10, 6))
            pos = nx.spring_layout(G)
            nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', arrows=True)
            nx.draw_networkx_edge_labels(G, pos, edge_labels={(u, v): f'Driver {d}' for u, v, d in G.edges(data="driver")})
            plt.title('Driver Transitions Across Teams')
            st.pyplot(plt)
    elif feature_name == "Team Performance Comparison":
        #import streamlit as st
        import pandas as pd
        import matplotlib.pyplot as plt
        import seaborn as sns

        # Load Data
        results = load_data('DPL_Datasets/results.csv')
        races = load_data('DPL_Datasets/races.csv')
        circuits = load_data('DPL_Datasets/circuits.csv')
        constructors = load_data('DPL_Datasets/constructors.csv')

        st.title("🏎️ Team Success Comparison")

        # Team selection
        team_options = constructors['constructorId'].unique()
        team1_id = st.selectbox("Select Primary Team:", team_options)
        opponent_ids = st.multiselect("Select Opponent Teams:", team_options, default=[team_options[0]])

        # Year range selection
        year_start, year_end = st.slider("Select Year Range:", min_value=int(races['year'].min()), max_value=int(races['year'].max()), value=(2010, 2023))

        def compare_team_success(team1_id, opponent_ids, year_start, year_end, races, results, circuits, constructors):

            # Ensure circuitId exists before merging
            if 'circuitId' not in races.columns:
                raise KeyError("'circuitId' is missing from races DataFrame. Check dataset structure.")

            # Get team names
            team1_name = constructors.loc[constructors['constructorId'] == team1_id, 'name'].values[0]
            opponent_names = {team_id: constructors.loc[constructors['constructorId'] == team_id, 'name'].values[0] for team_id in opponent_ids}

            # Merge race info with circuits if circuitId exists
            race_info = races.merge(circuits[['circuitId', 'name']], on='circuitId', how='left') if 'name' in circuits.columns else races.copy()

            # Merge results with race information, ensuring circuitId is correctly referenced
            full_data = results.merge(race_info[['raceId', 'year', 'circuitId']], on='raceId', how='left')

            # Check the column names to determine the correct circuitId column
            if 'circuitId_x' in full_data.columns:
                full_data.rename(columns={'circuitId_x': 'circuitId'}, inplace=True)
            elif 'circuitId_y' in full_data.columns:
                full_data.rename(columns={'circuitId_y': 'circuitId'}, inplace=True)
            # Proceed with filtering the data
            mask = (
                full_data['year'].between(year_start, year_end) & 
                full_data['constructorId'].isin([team1_id] + opponent_ids)
            )

            analysis_data = full_data[mask].copy()


            # Overall success rate (without circuit consideration)
            overall_success = []
            teams = [team1_id] + opponent_ids
            team_names = [team1_name] + list(opponent_names.values())

            for team_id, team_name in zip(teams, team_names):
                wins = len(analysis_data[(analysis_data['constructorId'] == team_id) & (analysis_data['positionOrder'] == 1)])
                total_races = len(analysis_data[analysis_data['constructorId'] == team_id])
                win_rate = (wins / total_races * 100) if total_races > 0 else 0
                overall_success.append({'Team': team_name, 'Wins': wins, 'Total Races': total_races, 'Win Rate': win_rate})

            overall_success_df = pd.DataFrame(overall_success)

            # Circuit-specific success rates
            circuit_success = []
            if 'name' in race_info.columns:
                analysis_data = analysis_data.merge(race_info[['raceId', 'name']], on='raceId', how='left')

            for circuit_id in analysis_data['circuitId'].dropna().unique():
                circuit_data = analysis_data[analysis_data['circuitId'] == circuit_id]
                circuit_name = circuit_data['name'].iloc[0] if 'name' in circuit_data.columns else f'Circuit {circuit_id}'

                circuit_entry = {'Circuit': circuit_name}
                for team_id, team_name in zip(teams, team_names):
                    team_wins = len(circuit_data[(circuit_data['constructorId'] == team_id) & (circuit_data['positionOrder'] == 1)])
                    team_races = len(circuit_data[circuit_data['constructorId'] == team_id])
                    circuit_entry[f'{team_name} Win Rate'] = (team_wins / team_races * 100) if team_races > 0 else 0

                circuit_success.append(circuit_entry)

            circuit_success_df = pd.DataFrame(circuit_success)

            # Create visualizations
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

            # Overall success rates
            ax1.bar(overall_success_df['Team'], overall_success_df['Win Rate'])
            ax1.set_title('Overall Win Rates')
            ax1.set_ylabel('Win Rate (%)')

            # Circuit-specific comparison
            circuit_comparison = pd.melt(circuit_success_df, id_vars=['Circuit'], var_name='Team', value_name='Win Rate')
            sns.boxplot(data=circuit_comparison, x='Team', y='Win Rate', ax=ax2)
            ax2.set_title('Win Rates Distribution Across Circuits')
            ax2.set_ylabel('Win Rate (%)')

            plt.tight_layout()

            
            return overall_success_df, circuit_success_df, fig

        # Run analysis on button click
        if st.button("📊 Compare Teams"):
            overall_df, circuit_df, fig = compare_team_success(team1_id, opponent_ids, year_start, year_end, races, results, circuits, constructors)

            st.subheader("🏆 Overall Win Rates")
            st.dataframe(overall_df)

            st.subheader("📍 Circuit-Specific Performance")
            st.dataframe(circuit_df)

            st.pyplot(fig)
    elif feature_name =="Driver Consistency in Race Performance":
        #import streamlit as st
        import pandas as pd
        import plotly.express as px

        # Load data
        results = load_data('DPL_Datasets/results.csv')
        drivers = load_data('DPL_Datasets/drivers.csv')

        st.title("🏎️ Driver Consistency Analysis")

        # Merge race results with driver details
        driver_consistency = results.merge(drivers, on='driverId', how='left')

        # Aggregate driver performance
        driver_consistency = driver_consistency.groupby(['driverId', 'forename', 'surname']).agg(
            avg_position=('positionOrder', 'mean'),
            position_std=('positionOrder', 'std'),
            total_races=('raceId', 'count')
        ).reset_index()

        # Define consistency score (lower std deviation = more consistency)
        driver_consistency['consistency_score'] = 1 / (driver_consistency['position_std'] + 1) * 100

        # Identify most consistent and fluctuating drivers
        consistent_drivers = driver_consistency.sort_values(by='consistency_score', ascending=False).head(10)
        fluctuating_drivers = driver_consistency.sort_values(by='position_std', ascending=False).head(10)

        # Display data tables
        st.subheader("📊 Most Consistent Drivers")
        st.dataframe(consistent_drivers[['forename', 'surname', 'avg_position', 'position_std', 'total_races', 'consistency_score']])

        st.subheader("⚡ Most Fluctuating Drivers")
        st.dataframe(fluctuating_drivers[['forename', 'surname', 'avg_position', 'position_std', 'total_races']])

        # Plotly visualizations
        fig_consistent = px.bar(consistent_drivers, x='surname', y='consistency_score', color='avg_position',
                                title='Most Consistent Drivers in Race Performance',
                                labels={'surname': 'Driver', 'consistency_score': 'Consistency Score'},
                                text_auto=True, template='plotly_dark')

        fig_fluctuating = px.bar(fluctuating_drivers, x='surname', y='position_std', color='avg_position',
                                title='Most Fluctuating Drivers in Race Performance',
                                labels={'surname': 'Driver', 'position_std': 'Position Standard Deviation'},
                                text_auto=True, template='plotly_dark')

        st.plotly_chart(fig_consistent)
        st.plotly_chart(fig_fluctuating)

    elif feature_name=="Lap Time Efficiency Analysis":
        #import streamlit as st
        import pandas as pd
        import plotly.express as px

        # Load datasets
        lap_times = load_data('DPL_Datasets/lap_times.csv')
        results = load_data('DPL_Datasets/results.csv')
        races = load_data('DPL_Datasets/races.csv')
        constructors = load_data('DPL_Datasets/constructors.csv')
        circuits = load_data('DPL_Datasets/circuits.csv') 

        st.title("⏱️ Lap Time Efficiency Analysis")

        # Merge lap times with race and constructor details
        lap_efficiency = lap_times.merge(results[['raceId', 'driverId', 'constructorId']], on=['raceId', 'driverId'], how='left')
        lap_efficiency = lap_efficiency.merge(races[['raceId', 'circuitId']], on='raceId', how='left')
        lap_efficiency = lap_efficiency.merge(constructors[['constructorId', 'name']], on='constructorId', how='left')
        lap_efficiency = lap_efficiency.merge(circuits[['circuitId', 'name']], on='circuitId', how='left', suffixes=('_team', '_circuit'))

        # Aggregate lap time efficiency per team and circuit
        lap_efficiency = lap_efficiency.groupby(['name_team', 'name_circuit']).agg(
            avg_lap_time=('milliseconds', 'mean'),
            min_lap_time=('milliseconds', 'min'),
            max_lap_time=('milliseconds', 'max')
        ).reset_index()

        # Calculate efficiency score
        lap_efficiency['efficiency_score'] = 1 / lap_efficiency['avg_lap_time'] * 1e6

        # Identify top-performing teams
        best_teams = lap_efficiency.sort_values(by='efficiency_score', ascending=False).head(10)

        st.subheader("🏆 Top Teams Maximizing Lap Time Efficiency")
        st.dataframe(best_teams[['name_team', 'name_circuit', 'avg_lap_time', 'efficiency_score']])

        # Visualization
        fig = px.bar(best_teams, x='name_team', y='efficiency_score', color='name_circuit',
                    title='Top Teams Maximizing Lap Time Efficiency',
                    labels={'name_team': 'Team', 'efficiency_score': 'Efficiency Score'},
                    text_auto=True, template='plotly_dark')

        st.plotly_chart(fig)
    
    elif feature_name =="Best Team Lineup":
        #import streamlit as st
        import pandas as pd
        import plotly.express as px

        # Load datasets
        driver_standings = load_data('DPL_Datasets/driver_standings.csv')
        drivers = load_data('DPL_Datasets/drivers.csv')
        st.title("🏎️ Best Team Lineup Analysis")

        # Merge driver standings with driver details
        driver_performance = driver_standings.merge(drivers, on='driverId', how='left')

        # Aggregate performance metrics
        driver_performance = driver_performance.groupby(['driverId', 'forename', 'surname', 'nationality']).agg(
            total_points=('points', 'sum'),
            avg_position=('position', 'mean'),
            total_wins=('wins', 'sum')
        ).reset_index()

        # Normalize performance scores
        driver_performance['performance_score'] = (
            driver_performance['total_points'] * 0.5 +
            (1 / driver_performance['avg_position']) * 50 +
            driver_performance['total_wins'] * 2
        )

        # Select top drivers for the best lineup
        best_drivers = driver_performance.sort_values(by='performance_score', ascending=False).head(2)

        st.subheader("🏆 Best Team Lineup Based on Performance")
        st.dataframe(best_drivers[['forename', 'surname', 'nationality', 'total_points', 'avg_position', 'total_wins', 'performance_score']])

        # Visualization
        fig = px.bar(best_drivers, x='surname', y='performance_score', color='nationality',
                    title='Top Performing Drivers for Best Team Lineup',
                    labels={'surname': 'Driver', 'performance_score': 'Performance Score'},
                    text_auto=True, template='plotly_dark')

        st.plotly_chart(fig)

    elif feature_name=="Predictions for 2025 Season":
        #import streamlit as st
        import pandas as pd
        import numpy as np
        import plotly.express as px
        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.preprocessing import LabelEncoder

        # Load your datasets (ensure they are preprocessed correctly)
        driver_standings = load_data('DPL_Datasets/driver_standings.csv')
        constructor_standings = load_data('DPL_Datasets/constructor_standings.csv')
        races = load_data('DPL_Datasets/races.csv')
        st.title("🏎️ 2025 Formula 1 Season Predictions")
        st.write("Based on historical data and machine learning models, let's predict the 2025 Driver and Constructor Champions!")

        # Merge required datasets
        driver_performance = driver_standings.merge(
            races[['raceId', 'year']], on='raceId', how='left'
        )
        constructor_performance = constructor_standings.merge(
            races[['raceId', 'year']], on='raceId', how='left'
        )

        # Aggregate historical performance
        driver_performance = driver_performance.groupby(['driverId', 'year']).agg(
            total_points=('points', 'sum'),
            avg_position=('position', 'mean'),
            total_wins=('wins', 'sum')
        ).reset_index()

        constructor_performance = constructor_performance.groupby(['constructorId', 'year']).agg(
            total_points=('points', 'sum'),
            avg_position=('position', 'mean'),
            total_wins=('wins', 'sum')
        ).reset_index()

        # Encode categorical data
        le_driver = LabelEncoder()
        driver_performance['driverId_encoded'] = le_driver.fit_transform(driver_performance['driverId'].astype(str))
        le_constructor = LabelEncoder()
        constructor_performance['constructorId_encoded'] = le_constructor.fit_transform(constructor_performance['constructorId'].astype(str))

        # Train models
        X_driver = driver_performance[['driverId_encoded', 'year', 'total_points', 'avg_position', 'total_wins']]
        y_driver = driver_performance['total_points']
        X_train, X_test, y_train, y_test = train_test_split(X_driver, y_driver, test_size=0.2, random_state=42)
        driver_model = RandomForestRegressor(n_estimators=100, random_state=42)
        driver_model.fit(X_train, y_train)

        X_constructor = constructor_performance[['constructorId_encoded', 'year', 'total_points', 'avg_position', 'total_wins']]
        y_constructor = constructor_performance['total_points']
        X_train, X_test, y_train, y_test = train_test_split(X_constructor, y_constructor, test_size=0.2, random_state=42)
        constructor_model = RandomForestRegressor(n_estimators=100, random_state=42)
        constructor_model.fit(X_train, y_train)

        # Predict champions for 2025
        future_driver = driver_performance.groupby('driverId').last().reset_index()
        future_driver['year'] = 2025
        future_driver['driverId_encoded'] = le_driver.transform(future_driver['driverId'].astype(str))
        future_driver['predicted_points'] = driver_model.predict(
            future_driver[['driverId_encoded', 'year', 'total_points', 'avg_position', 'total_wins']]
        )

        future_constructor = constructor_performance.groupby('constructorId').last().reset_index()
        future_constructor['year'] = 2025
        future_constructor['constructorId_encoded'] = le_constructor.transform(future_constructor['constructorId'].astype(str))
        future_constructor['predicted_points'] = constructor_model.predict(
            future_constructor[['constructorId_encoded', 'year', 'total_points', 'avg_position', 'total_wins']]
        )

        # Identify predicted winners
        driver_winner = future_driver.sort_values(by='predicted_points', ascending=False).iloc[0]
        constructor_winner = future_constructor.sort_values(by='predicted_points', ascending=False).iloc[0]

        # Display Results
        st.subheader("🏆 Predicted 2025 Drivers' Champion")
        st.write(f"**Driver ID:** {driver_winner['driverId']}")
        st.write(f"**Predicted Points:** {driver_winner['predicted_points']:.2f}")

        st.subheader("🏆 Predicted 2025 Constructors' Champion")
        st.write(f"**Constructor ID:** {constructor_winner['constructorId']}")
        st.write(f"**Predicted Points:** {constructor_winner['predicted_points']:.2f}")

        # Visualization: Top 10 Predicted Drivers
        top_drivers = future_driver.sort_values(by='predicted_points', ascending=False).head(10)
        fig_drivers = px.bar(top_drivers, x='driverId', y='predicted_points', title="Top 10 Predicted Drivers (2025)", labels={'driverId': 'Driver ID', 'predicted_points': 'Predicted Points'}, text_auto=True, template='plotly_dark')
        st.plotly_chart(fig_drivers)

        # Visualization: Top 10 Predicted Constructors
        top_constructors = future_constructor.sort_values(by='predicted_points', ascending=False).head(10)
        fig_constructors = px.bar(top_constructors, x='constructorId', y='predicted_points', title="Top 10 Predicted Constructors (2025)", labels={'constructorId': 'Constructor ID', 'predicted_points': 'Predicted Points'}, text_auto=True, template='plotly_dark')
        st.plotly_chart(fig_constructors)
    
    elif feature_name == "Struggling Teams Analysis":
        #import streamlit as st
        import pandas as pd
        import numpy as np
        import plotly.express as px
        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import LabelEncoder

        # Load data
        races = load_data('DPL_Datasets/races.csv')
        constructor_standings = load_data('DPL_Datasets/constructor_standings.csv')

        st.title("⚠️ Struggling Teams Analysis (2025 Prediction)")
        st.write("Predicting which teams are most likely to underperform in the 2025 season based on historical trends.")

        # Merge required datasets
        team_performance = constructor_standings.merge(
            races[['raceId', 'year']], on='raceId', how='left'
        )

        # Calculate historical performance trends
        team_performance = team_performance.groupby(['constructorId', 'year']).agg(
            total_points=('points', 'sum'),
            avg_position=('position', 'mean'),
            total_wins=('wins', 'sum')
        ).reset_index()

        # Calculate performance decline indicators
        team_performance['points_change'] = team_performance.groupby('constructorId')['total_points'].diff()
        team_performance['position_change'] = team_performance.groupby('constructorId')['avg_position'].diff()
        team_performance['win_change'] = team_performance.groupby('constructorId')['total_wins'].diff()

        # Define struggling teams (teams with declining points, worse positions, fewer wins)
        team_performance['struggling'] = (
            (team_performance['points_change'] < 0) &
            (team_performance['position_change'] > 0) &
            (team_performance['win_change'] < 0)
        ).astype(int)

        # Encode constructor IDs
        le_team = LabelEncoder()
        team_performance['constructorId_encoded'] = le_team.fit_transform(team_performance['constructorId'].astype(str))

        # Train a predictive model
        X = team_performance[['constructorId_encoded', 'year', 'total_points', 'avg_position', 'total_wins']]
        y = team_performance['struggling']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Predict struggling teams for 2025
        future_teams = team_performance.groupby('constructorId').last().reset_index()
        future_teams['year'] = 2025
        future_teams['constructorId_encoded'] = le_team.transform(future_teams['constructorId'].astype(str))
        future_teams['predicted_struggling'] = model.predict(
            future_teams[['constructorId_encoded', 'year', 'total_points', 'avg_position', 'total_wins']]
        )

        # Identify struggling teams
        struggling_teams = future_teams[future_teams['predicted_struggling'] == 1]

        st.subheader("🔻 Predicted Struggling Teams for 2025")
        if struggling_teams.empty:
            st.write("No teams are predicted to struggle significantly in 2025.")
        else:
            st.dataframe(struggling_teams[['constructorId', 'total_points', 'avg_position', 'total_wins']])

        # Visualization: Struggling Teams
        fig = px.bar(
            struggling_teams, 
            x='constructorId', 
            y='total_points', 
            title="Teams Likely to Struggle in 2025", 
            labels={'constructorId': 'Constructor', 'total_points': 'Total Points'},
            text_auto=True, 
            template='plotly_dark'
        )
        st.plotly_chart(fig)

    elif feature_name =="Driver-Specific Track Struggles":
        #import streamlit as st
        import pandas as pd
        import plotly.express as px

        # Load data
        results = load_data('DPL_Datasets/results.csv')
        races = load_data('DPL_Datasets/races.csv')
        circuits = load_data('DPL_Datasets/circuits.csv')

        st.title("🏁 Driver-Specific Track Struggles & Excellence")
        st.write("Analyze circuits where F1 drivers consistently struggle or excel based on historical race performance.")

        # Merge required datasets
        results = results.merge(races[['raceId', 'circuitId', 'year']], on='raceId', how='left')
        results = results.merge(circuits[['circuitId', 'name']], on='circuitId', how='left')

        # Define struggle and excellence based on finishing position
        median_position = results['positionOrder'].median()
        results['struggle'] = results['positionOrder'] > median_position
        results['excel'] = results['positionOrder'] <= median_position

        # Group data to analyze performance trends per driver per circuit
        driver_circuit_performance = results.groupby(['driverId', 'name']).agg(
            total_races=('raceId', 'count'),
            struggle_count=('struggle', 'sum'),
            excel_count=('excel', 'sum')
        ).reset_index()

        # Calculate struggle and excellence percentages
        driver_circuit_performance['struggle_pct'] = (
            driver_circuit_performance['struggle_count'] / driver_circuit_performance['total_races'] * 100
        )
        driver_circuit_performance['excel_pct'] = (
            driver_circuit_performance['excel_count'] / driver_circuit_performance['total_races'] * 100
        )

        # Identify drivers who struggle or excel the most on specific circuits
        struggle_tracks = driver_circuit_performance.sort_values(by='struggle_pct', ascending=False)
        excel_tracks = driver_circuit_performance.sort_values(by='excel_pct', ascending=False)

        # Dropdown for driver selection
        driver_list = driver_circuit_performance['driverId'].unique()
        selected_driver = st.selectbox("Select a Driver", driver_list)

        # Filter data for the selected driver
        driver_performance = driver_circuit_performance[driver_circuit_performance['driverId'] == selected_driver]

        # Display struggle & excellence stats
        st.subheader(f"📊 Performance Summary for Driver {selected_driver}")
        st.write(driver_performance[['name', 'total_races', 'struggle_pct', 'excel_pct']])

        # Visualization: Scatter plot of struggle vs excellence across circuits
        fig = px.scatter(driver_circuit_performance, x='struggle_pct', y='excel_pct',
                        color='name', hover_data=['driverId'],
                        title='Driver Performance Across Circuits',
                        labels={'struggle_pct': 'Struggle Percentage', 'excel_pct': 'Excellence Percentage'},
                        template='plotly_dark')
        st.plotly_chart(fig)

        # Bar chart of the top struggling circuits
        fig_struggle = px.bar(struggle_tracks.head(10), x='name', y='struggle_pct',
                            title="Top 10 Tracks Where Drivers Struggle the Most",
                            labels={'name': 'Circuit', 'struggle_pct': 'Struggle %'},
                            text_auto=True, template='plotly_dark')
        st.plotly_chart(fig_struggle)

        # Bar chart of the top excelling circuits
        fig_excel = px.bar(excel_tracks.head(10), x='name', y='excel_pct',
                        title="Top 10 Tracks Where Drivers Excel the Most",
                        labels={'name': 'Circuit', 'excel_pct': 'Excellence %'},
                        text_auto=True, template='plotly_dark')
        st.plotly_chart(fig_excel)

    elif feature_name == "Championship Retention Probability":

        import pandas as pd
        import plotly.express as px

        # Load data
        driver_standings = load_data('DPL_Datasets/driver_standings.csv')
        races = load_data('DPL_Datasets/races.csv')

        st.title("🏆 Championship Retention Probability")
        st.write("Analyze the likelihood of a reigning F1 champion defending their title in the next season.")

        # Merge required datasets
        driver_standings = driver_standings.merge(races[['raceId', 'year']], on='raceId', how='left')

        # Filter only championship winners (position == 1)
        champions = driver_standings[driver_standings['position'] == 1].copy()

        # Sort by year to check for consecutive wins
        champions = champions.sort_values(by=['driverId', 'year'])

        # Identify back-to-back champions
        champions['retained_title'] = champions['driverId'].eq(champions['driverId'].shift(1))

        # Calculate probability of retaining the title
        retention_probability = champions['retained_title'].mean()

        # Analyze historical trends by decade
        champions['decade'] = (champions['year'] // 10) * 10
        retention_by_decade = champions.groupby('decade')['retained_title'].mean().reset_index()

        # Display overall retention probability
        st.subheader("📊 Overall Retention Probability")
        st.write(f"🔹 The probability of an F1 champion defending their title in the next season is **{retention_probability:.2%}**.")

        # Display historical retention trends
        st.subheader("📉 Retention Probability by Decade")
        st.dataframe(retention_by_decade)

        # Interactive Visualization: Retention Trends Over Decades
        fig = px.bar(retention_by_decade, x='decade', y='retained_title',
                    labels={'decade': 'Decade', 'retained_title': 'Retention Probability'},
                    title='Championship Retention Probability Over Decades',
                    text_auto=True, color='retained_title', template='plotly_dark')

        st.plotly_chart(fig)
    
    elif feature_name == "Champion Age Trends":
        #import streamlit as st
        import pandas as pd
        import plotly.express as px

        # Load data
        driver_standings = load_data('DPL_Datasets/driver_standings.csv')
        races = load_data('DPL_Datasets/races.csv')
        drivers = load_data('DPL_Datasets/drivers.csv')

        st.title("🏆 Champion Age Trends")
        st.write("Analyze the age distribution of Formula 1 champions across different decades.")

        # Merge driver standings with drivers
        driver_standings = driver_standings.merge(drivers, on='driverId', how='left', suffixes=('', '_driver'))
        driver_standings = driver_standings.merge(races[['raceId', 'year']], on='raceId', how='left', suffixes=('', '_race'))

        # Filter only championship winners (position == 1)
        champions = driver_standings[driver_standings['position'] == 1].copy()

        # Convert birthdate to datetime and calculate age at win
        champions['dob'] = pd.to_datetime(champions['dob'])  
        champions['age_at_win'] = champions['year'] - champions['dob'].dt.year

        # Categorize by decade
        champions['decade'] = (champions['year'] // 10) * 10

        # Aggregate data to find age distribution per decade
        age_trends = champions.groupby('decade')['age_at_win'].describe().reset_index()

        # Display historical age trends
        st.subheader("📊 Champion Age Statistics by Decade")
        st.dataframe(age_trends)

        # Interactive Visualization: Age Distribution Across Decades
        fig = px.box(champions, x='decade', y='age_at_win',
                    labels={'decade': 'Decade', 'age_at_win': 'Champion Age'},
                    title='Champion Age Trends Across Decades',
                    color='decade', 
                    template='plotly_dark',  
                    boxmode='overlay')

        fig.update_traces(marker=dict(size=8, opacity=0.7, line=dict(width=1)))
        fig.update_layout(
            title_font_size=20,
            title_x=0.5,  
            xaxis_title_font_size=16,
            yaxis_title_font_size=16,
            font=dict(family='Arial', size=14),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )

        st.plotly_chart(fig)


    elif feature_name == "Bonus Challenge (Optional)":
        #import streamlit as st
        import pandas as pd
        import plotly.express as px
        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import LabelEncoder

        st.title("🔮 Predict Future F1 Driver Transfers")
        st.write("Analyze historical team changes to predict which team a driver might move to next.")

        # Load data
        results = load_data('DPL_Datasets/results.csv')
        races = load_data('DPL_Datasets/races.csv')
        
        # Merge required datasets
        driver_transfers = results[['driverId', 'constructorId', 'raceId']].merge(
            races[['raceId', 'year']], on='raceId', how='left'
        )

        # Sort data by year for transition analysis
        driver_transfers = driver_transfers.sort_values(by=['driverId', 'year'])

        # Identify team changes
        driver_transfers['prev_team'] = driver_transfers.groupby('driverId')['constructorId'].shift(1)
        driver_transfers['team_changed'] = driver_transfers['constructorId'] != driver_transfers['prev_team']

        # Fill missing values with a placeholder and convert to string
        driver_transfers['prev_team'] = driver_transfers['prev_team'].fillna(-1).astype(int).astype(str)

        # Count transitions per driver
        driver_transitions = driver_transfers.groupby('driverId').agg(
            total_transfers=('team_changed', 'sum'),
            last_team=('constructorId', 'last'),
            most_recent_year=('year', 'max')
        ).reset_index()

        # Encode categorical data
        le_team = LabelEncoder()
        all_teams = driver_transfers['constructorId'].astype(str).tolist() + ['-1']  # Include placeholder for first-time drivers
        le_team.fit(all_teams)

        driver_transfers['constructorId_encoded'] = le_team.transform(driver_transfers['constructorId'].astype(str))
        driver_transfers['prev_team_encoded'] = le_team.transform(driver_transfers['prev_team'])

        # Define features and target
        X = driver_transfers[['prev_team_encoded', 'year']]
        y = driver_transfers['constructorId_encoded']

        # Train a predictive model
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Predict future teams for drivers
        future_predictions = driver_transitions[['driverId', 'last_team', 'most_recent_year']].copy()
        future_predictions['next_year'] = future_predictions['most_recent_year'] + 1
        future_predictions['last_team'] = future_predictions['last_team'].fillna(-1).astype(int).astype(str)
        future_predictions['prev_team_encoded'] = le_team.transform(future_predictions['last_team'])

        # Rename 'next_year' to 'year' for consistency with training data
        future_predictions = future_predictions.rename(columns={'next_year': 'year'})

        future_predictions['predicted_team_encoded'] = model.predict(
            future_predictions[['prev_team_encoded', 'year']]
        )
        future_predictions['predicted_team'] = le_team.inverse_transform(future_predictions['predicted_team_encoded'])

        st.subheader("📌 Predicted Future Teams for Drivers")
        st.dataframe(future_predictions[['driverId', 'last_team', 'predicted_team']])

        # Interactive Visualization: Team Transfers Over Time
        fig = px.bar(future_predictions, x='driverId', y='predicted_team',
                    labels={'driverId': 'Driver ID', 'predicted_team': 'Predicted Team'},
                    title='Predicted Driver Transfers for Next Season',
                    text_auto=True, color='predicted_team', template='plotly_dark')

        st.plotly_chart(fig)


    st.markdown('</div>', unsafe_allow_html=True)
    col1, col2,col3 = st.columns(3)
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        with st.container():
            co1, co2, co3 = st.columns(3)
            with co2:
                if st.button("Back to Home", key="back_to_home2", help="Return to home page"):
                    st.session_state.page = "home"
                    st.rerun()

# Home page
def home_page():
    add_breadcrumb()
    
    # Header
    #st.markdown('<div class="main-header">', unsafe_allow_html=True)
    st.markdown('<h1 class="main-title" style="text-align: center;">🏎️ Formula 1 Data Explorer</h1>', unsafe_allow_html=True)
    #st.markdown('<p class="sub-title">Discover insights from Formula 1 data with interactive visualizations and analysis</p>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Create two main cards for dataset exploration and feature engineering
    st.markdown('<div class="card-container">', unsafe_allow_html=True)
    
    # Dataset exploration card
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            create_card(
                "📊 Dataset Exploration",
                "Explore comprehensive F1 datasets including races, drivers, constructors, and performance metrics. Analyze trends, statistics, and relationships between different data elements.",
                "Explore Datasets",
                "explore_datasets"
            ),
            unsafe_allow_html=True
        )
        st.markdown("<br>", unsafe_allow_html=True)
        with st.container():
            co1, co2, co3 = st.columns(3)
            with co2:
                st.button("Explore Datasets", key="explore_datasets", help="View available datasets", on_click=lambda: set_page("datasets"))


    
    # Feature engineering card
    with col2:
        st.markdown(
            create_card(
                "🔍 Feature Engineering Insights",
                "Discover deep insights through advanced feature engineering. Analyze driver performance, race strategies, head-to-head comparisons, and predictive models.",
                "View Insights",
                "view_insights"
            ),
            unsafe_allow_html=True
        )
        st.markdown("<br>", unsafe_allow_html=True)
        with st.container():
            co1, co2, co3 = st.columns(3)
            with co2:
                st.button("View Insights", key="view_insights", help="Explore feature engineering insights", on_click=lambda: set_page("features"))
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Recent updates and statistics section
    section_header("Recent Updates & Statistics")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="metric-container">
            <div class="metric-title">Last Data Update</div>
            <div class="metric-value">2023 Season</div>
            <div>Including all Grand Prix results</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-container">
            <div class="metric-title">Total Datasets</div>
            <div class="metric-value">14</div>
            <div>Comprehensive F1 statistics</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-container">
            <div class="metric-title">Featured Analysis</div>
            <div class="metric-value">Pit Stop Impact</div>
            <div>How strategies affect race outcomes</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Documentation section
    section_header("Quick Start Guide")
    st.markdown("""
    <div class="metric-container">
        <p>Welcome to the Formula 1 Data Explorer! This interactive application allows you to explore F1 data and discover insights through various analyses and visualizations. Here's how to get started:</p>
        <ol>
            <li><strong>Explore Datasets</strong>: View and analyze raw F1 data across multiple categories.</li>
            <li><strong>Feature Engineering Insights</strong>: Discover patterns and trends through specialized analyses.</li>
            <li><strong>Interactive Visualizations</strong>: All charts and graphs are interactive - hover, zoom, and explore.</li>
            <li><strong>Export Options</strong>: Download data and visualizations for your own use.</li>
        </ol>
        <p>Select one of the main options above to begin your exploration!</p>
    </div>
    """, unsafe_allow_html=True)

# Datasets page
def datasets_page():
    add_breadcrumb([("Home", "home"), ("Datasets", "datasets")])
    
    # Header
    st.markdown('<h1 class="main-title">Formula 1 Datasets</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Explore comprehensive data from the world of Formula 1</p>', unsafe_allow_html=True)
    
    # Dataset cards in a grid
    st.markdown('<div class="card-container">', unsafe_allow_html=True)
    
    # Create cards for each dataset
    dataset_descriptions = {
        "Circuits": "Information about F1 race circuits including location, country, and layout details.",
        "Constructor Results": "Results of constructors in each race including points and status.",
        "Constructor Standings": "Constructor championship standings after each race.",
        "Constructors": "Details about F1 teams (constructors) including nationality, founding year, and performance.",
        "Driver Standings": "Driver championship standings after each race.",
        "Drivers": "Comprehensive data about F1 drivers including nationality, date of birth, and career statistics.",
        "Lap Times": "Detailed lap time data for each driver in every race.",
        "Pit Stops": "Data on pit stops including duration, lap number, and driver/team information.",
        "Qualifying": "Qualifying results including position, lap time, and session details.",
        "Races": "Information about each F1 race including date, location, and circuit.",
        "Results": "Race results including finishing position, points, and status for each driver.",
        "Seasons": "Information about each F1 season including year and champion.",
        "Sprint Results": "Results of sprint races including finishing position and points.",
        "Status": "Status information for each race including reasons for non-finishes."
    }
    
    # Create a 3-column grid for dataset cards
    cols = st.columns(3)
    
    # Add cards for each dataset
    for i, (dataset, description) in enumerate(dataset_descriptions.items()):
        with cols[i % 3]:
            st.markdown(
                create_card(
                    f"{dataset}",
                    description,
                    "Explore Dataset",
                    f"dataset_btn_{dataset.replace(' ', '_')}"
                ),
                unsafe_allow_html=True
            )
            st.markdown("<br>", unsafe_allow_html=True)
            with st.container():
                co1, co2, co3 = st.columns(3)
                with co2:
                    if st.button(f"Explore Dataset", key=f"dataset_btn_{dataset.replace(' ', '_')}", help=f"View {dataset} dataset"):
                        st.session_state.page = f"dataset_{dataset}"
                        st.session_state.selected_dataset = dataset
                        st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)
    col1, col2,col3 = st.columns(3)
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        with st.container():
            co1, co2, co3 = st.columns(3)
            with co2:
                if st.button("Back to Home", key="back_to_home_datasets", help="Return to home page"):
                    st.session_state.page = "home"
                    st.rerun()

# Features page
def features_page():
    add_breadcrumb([("Home", "home"), ("Feature Engineering", "features")])
    
    # Header
    st.markdown('<h1 class="main-title">Feature Engineering Insights</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Discover patterns and insights through advanced analytics</p>', unsafe_allow_html=True)
    
    # Feature cards in a grid
    st.markdown('<div class="card-container">', unsafe_allow_html=True)
    
    # Create a 2-column grid for feature cards
    cols = st.columns(2)
    
    # Add cards for each feature
    for i, (feature, description) in enumerate(feature_sections.items()):
        with cols[i % 2]:
            st.markdown(
                create_card(
                    f"{feature}",
                    description,
                    "View Analysis",
                    f"feature_btn_{feature.replace(' ', '_').replace('&', 'and')}"
                ),
                unsafe_allow_html=True
            )
            
            st.markdown("<br>", unsafe_allow_html=True)
            with st.container():
                co1, co2, co3 = st.columns(3)
                with co2:
                    if st.button(f"View Analysis", key=f"feature_btn_{feature.replace(' ', '_').replace('&', 'and')}", help=f"View {feature} analysis"):
                        st.session_state.page = f"feature_{feature}"
                        st.session_state.selected_feature = feature
                        st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    col1, col2,col3 = st.columns(3)
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        with st.container():
            co1, co2, co3 = st.columns(3)
            with co2:
                if st.button("Back to Home", key="back_to_home_features", help="Return to home page"):
                    st.session_state.page = "home"
                    st.rerun()

# Function to set the current page
def set_page(page):
    st.session_state.page = page

# Main function with navigation management
def main():
    # Add CSS styling
    add_custom_css()
    add_animated_background()
    
    # Initialize session state for navigation
    if 'page' not in st.session_state:
        st.session_state.page = "home"
    
    # Handle navigation
    current_page = st.session_state.page
    
    # Route to the correct page
    if current_page == "home":
        home_page()
    elif current_page == "datasets":
        datasets_page()
    elif current_page == "features":
        features_page()
    elif current_page.startswith("dataset_"):
        dataset_name = st.session_state.selected_dataset
        dataset_dashboard(dataset_name)
    elif current_page.startswith("feature_"):
        feature_name = st.session_state.selected_feature
        feature_dashboard(feature_name)

# Run the app
if __name__ == "__main__":
    main()