import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="TrafficFlow AI",
    page_icon="🚦",
    layout="wide"
)


# =========================================================
# LIGHT YELLOW THEME
# =========================================================

st.markdown(
    """
    <style>
    /* Light yellow background */
    .stApp {
        background-color: #fffbea;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #fff3b0;
    }

    /* Main text */
    h1, h2, h3 {
        color: #5c4300;
    }

    /* Buttons */
    .stButton > button {
        background-color: #f4c430;
        color: #3d2b00;
        border-radius: 10px;
        border: none;
        font-weight: bold;
    }

    .stButton > button:hover {
        background-color: #e6b800;
        color: white;
    }

    /* Metric styling */
    [data-testid="stMetric"] {
        background-color: #fff8d6;
        border-radius: 12px;
        padding: 15px;
        border: 1px solid #f0d66b;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# LOAD DATASET
# =========================================================

@st.cache_data
def load_dataset():

    data = pd.read_csv("traffic_volume.csv")

    data["date_time"] = pd.to_datetime(
        data["date_time"],
        format="%d-%m-%Y %H:%M:%S",
        errors="coerce"
    )

    data["year"] = data["date_time"].dt.year

    data["month"] = data["date_time"].dt.month

    data["day"] = data["date_time"].dt.day

    data["hour"] = data["date_time"].dt.hour

    data["day_of_week"] = (
        data["date_time"].dt.dayofweek
    )

    data["is_weekend"] = data[
        "day_of_week"
    ].apply(
        lambda x: 1 if x >= 5 else 0
    )

    return data


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():

    model = joblib.load(
        "traffic_model.pkl"
    )

    feature_columns = joblib.load(
        "model_features.pkl"
    )

    return model, feature_columns


# =========================================================
# LOAD EVERYTHING
# =========================================================

try:

    df = load_dataset()

    model, feature_columns = load_model()

except Exception as e:

    st.error("⚠️ Unable to load the required files.")

    st.info(
        """
        Make sure these files are in the same folder as app.py:

        • traffic_volume.csv
        • traffic_model.pkl
        • model_features.pkl
        """
    )

    st.stop()


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("🚦 TrafficFlow AI")

st.sidebar.write(
    "Traffic Volume Prediction System"
)

st.sidebar.divider()

page = st.sidebar.radio(
    "Choose Page",
    [
        "🏠 Dashboard",
        "🔮 Prediction",
        "📊 Analytics",
        "🤖 Model Information"
    ]
)

st.sidebar.divider()

st.sidebar.info(
    """
    🚗 Traffic Volume Prediction

    Machine Learning Project

    Built with:
    Python
    Pandas
    Scikit-learn
    Plotly
    Streamlit
    """
)


# =========================================================
# DASHBOARD PAGE
# =========================================================

if page == "🏠 Dashboard":

    st.title("🚦 TrafficFlow AI")

    st.subheader(
        "Intelligent Traffic Volume Prediction using Machine Learning"
    )

    st.write(
        """
        This application predicts traffic volume using
        weather conditions and time-related information.
        """
    )

    st.divider()

    # =====================================================
    # DASHBOARD STATISTICS
    # =====================================================

    average_traffic = df[
        "traffic_volume"
    ].mean()

    maximum_traffic = df[
        "traffic_volume"
    ].max()

    minimum_traffic = df[
        "traffic_volume"
    ].min()

    average_temperature = df[
        "temp"
    ].mean()

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "🚗 Average Traffic",
            f"{average_traffic:,.0f}"
        )

    with col2:

        st.metric(
            "🚦 Maximum Traffic",
            f"{maximum_traffic:,.0f}"
        )

    with col3:

        st.metric(
            "📉 Minimum Traffic",
            f"{minimum_traffic:,.0f}"
        )

    with col4:

        st.metric(
            "🌡️ Average Temperature",
            f"{average_temperature:.1f}"
        )

    st.divider()

    # =====================================================
    # TRAFFIC BY HOUR
    # =====================================================

    st.header("📈 Traffic Volume by Hour")

    hourly_traffic = (
        df.groupby("hour")[
            "traffic_volume"
        ]
        .mean()
        .reset_index()
    )

    fig_hour = px.line(
        hourly_traffic,
        x="hour",
        y="traffic_volume",
        markers=True,
        title="Average Traffic Volume by Hour"
    )

    fig_hour.update_layout(
        height=450,
        template="plotly_white",
        paper_bgcolor="#fffbea",
        plot_bgcolor="#fffbea"
    )

    st.plotly_chart(
        fig_hour,
        use_container_width=True
    )

    # =====================================================
    # TEMPERATURE + CLOUDS
    # =====================================================

    col1, col2 = st.columns(2)

    with col1:

        st.subheader(
            "🌡️ Temperature vs Traffic"
        )

        fig_temp = px.scatter(
            df,
            x="temp",
            y="traffic_volume",
            title="Temperature vs Traffic",
            opacity=0.65
        )

        fig_temp.update_layout(
            height=400,
            template="plotly_white",
            paper_bgcolor="#fffbea",
            plot_bgcolor="#fffbea"
        )

        st.plotly_chart(
            fig_temp,
            use_container_width=True
        )

    with col2:

        st.subheader(
            "☁️ Cloud Coverage vs Traffic"
        )

        fig_cloud = px.scatter(
            df,
            x="clouds_all",
            y="traffic_volume",
            title="Cloud Coverage vs Traffic",
            opacity=0.65
        )

        fig_cloud.update_layout(
            height=400,
            template="plotly_white",
            paper_bgcolor="#fffbea",
            plot_bgcolor="#fffbea"
        )

        st.plotly_chart(
            fig_cloud,
            use_container_width=True
        )

    # =====================================================
    # WEATHER GRAPH
    # =====================================================

    st.header("🌦️ Average Traffic by Weather")

    weather_traffic = (
        df.groupby("weather_main")[
            "traffic_volume"
        ]
        .mean()
        .reset_index()
    )

    fig_weather = px.bar(
        weather_traffic,
        x="weather_main",
        y="traffic_volume",
        title="Average Traffic by Weather"
    )

    fig_weather.update_layout(
        height=450,
        template="plotly_white",
        paper_bgcolor="#fffbea",
        plot_bgcolor="#fffbea"
    )

    st.plotly_chart(
        fig_weather,
        use_container_width=True
    )


# =========================================================
# PREDICTION PAGE
# =========================================================

elif page == "🔮 Prediction":

    st.title("🔮 Traffic Prediction")

    st.write(
        "Enter weather and time information to predict traffic volume."
    )

    st.divider()

    # =====================================================
    # TIME INFORMATION
    # =====================================================

    st.header("📅 Time Information")

    col1, col2, col3 = st.columns(3)

    with col1:

        year = st.number_input(
            "Year",
            min_value=2018,
            max_value=2035,
            value=2018
        )

        month = st.number_input(
            "Month",
            min_value=1,
            max_value=12,
            value=1
        )

    with col2:

        day = st.number_input(
            "Day",
            min_value=1,
            max_value=31,
            value=3
        )

        hour = st.slider(
            "Hour",
            min_value=0,
            max_value=23,
            value=8
        )

    with col3:

        days = [
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
            "Sunday"
        ]

        day_of_week = st.selectbox(
            "Day of Week",
            days
        )

        day_number = days.index(
            day_of_week
        )

        is_weekend = (
            1
            if day_number >= 5
            else 0
        )

    st.divider()

    # =====================================================
    # WEATHER INFORMATION
    # =====================================================

    st.header("🌦️ Weather Information")

    col1, col2 = st.columns(2)

    with col1:

        temperature = st.number_input(
            "🌡️ Temperature",
            value=float(
                df["temp"].mean()
            )
        )

        rainfall = st.number_input(
            "🌧️ Rainfall (last 1 hour)",
            min_value=0.0,
            value=0.0
        )

        snowfall = st.number_input(
            "❄️ Snowfall (last 1 hour)",
            min_value=0.0,
            value=0.0
        )

    with col2:

        clouds = st.slider(
            "☁️ Cloud Coverage (%)",
            min_value=0,
            max_value=100,
            value=50
        )

        weather_main_values = sorted(
            df[
                "weather_main"
            ]
            .dropna()
            .unique()
            .tolist()
        )

        weather_description_values = sorted(
            df[
                "weather_description"
            ]
            .dropna()
            .unique()
            .tolist()
        )

        weather_main = st.selectbox(
            "🌦️ Weather",
            weather_main_values
        )

        weather_description = st.selectbox(
            "📝 Weather Description",
            weather_description_values
        )

    st.divider()

    # =====================================================
    # PREDICT BUTTON
    # =====================================================

    predict_button = st.button(
        "🚦 Predict Traffic Volume",
        use_container_width=True
    )

    if predict_button:

        # =================================================
        # CREATE INPUT DATA
        # =================================================

        input_data = pd.DataFrame(
            0,
            index=[0],
            columns=feature_columns
        )

        # =================================================
        # NUMERICAL FEATURES
        # =================================================

        if "temp" in input_data.columns:

            input_data["temp"] = temperature

        if "rain_1h" in input_data.columns:

            input_data["rain_1h"] = rainfall

        if "snow_1h" in input_data.columns:

            input_data["snow_1h"] = snowfall

        if "clouds_all" in input_data.columns:

            input_data["clouds_all"] = clouds

        if "year" in input_data.columns:

            input_data["year"] = year

        if "month" in input_data.columns:

            input_data["month"] = month

        if "day" in input_data.columns:

            input_data["day"] = day

        if "hour" in input_data.columns:

            input_data["hour"] = hour

        if "day_of_week" in input_data.columns:

            input_data[
                "day_of_week"
            ] = day_number

        if "is_weekend" in input_data.columns:

            input_data[
                "is_weekend"
            ] = is_weekend

        # =================================================
        # WEATHER MAIN
        # =================================================

        weather_main_column = (
            "weather_main_"
            + weather_main
        )

        if (
            weather_main_column
            in input_data.columns
        ):

            input_data[
                weather_main_column
            ] = 1

        # =================================================
        # WEATHER DESCRIPTION
        # =================================================

        weather_description_column = (
            "weather_description_"
            + weather_description
        )

        if (
            weather_description_column
            in input_data.columns
        ):

            input_data[
                weather_description_column
            ] = 1

        # =================================================
        # PREDICTION
        # =================================================

        prediction = model.predict(
            input_data
        )[0]

        prediction = max(
            0,
            prediction
        )

        st.divider()

        st.subheader(
            "🚦 Prediction Result"
        )

        st.success(
            f"Predicted Traffic Volume: "
            f"{prediction:,.0f} vehicles"
        )

        # =================================================
        # TRAFFIC LEVEL
        # =================================================

        average_traffic = df["traffic_volume"].mean()
        if prediction < average_traffic * 0.75:

            st.success(
                "🟢 Traffic Level: LOW"
            )

        elif prediction < average_traffic * 1.25:

            st.warning(
                "🟡 Traffic Level: MODERATE"
            )

        else:

            st.error(
                "🔴 Traffic Level: HIGH"
            )

        # =================================================
        # PREDICTION DETAILS
        # =================================================

        st.header("📋 Prediction Details")

        result_data = pd.DataFrame({

            "Parameter": [

                "Year",
                "Month",
                "Day",
                "Hour",
                "Day of Week",
                "Temperature",
                "Rainfall",
                "Snowfall",
                "Cloud Coverage",
                "Weather",
                "Weather Description"

            ],

            "Value": [

                year,
                month,
                day,
                hour,
                day_of_week,
                temperature,
                rainfall,
                snowfall,
                clouds,
                weather_main,
                weather_description

            ]

        })

        st.dataframe(
            result_data,
            use_container_width=True,
            hide_index=True
        )


# =========================================================
# ANALYTICS PAGE
# =========================================================

elif page == "📊 Analytics":

    st.title("📊 Traffic Analytics")

    st.write(
        "Explore traffic patterns and relationships in the dataset."
    )

    st.divider()

    # =====================================================
    # TRAFFIC BY HOUR
    # =====================================================

    st.header(
        "🕐 Average Traffic Volume by Hour"
    )

    hourly_data = (
        df.groupby("hour")[
            "traffic_volume"
        ]
        .mean()
        .reset_index()
    )

    fig1 = px.line(
        hourly_data,
        x="hour",
        y="traffic_volume",
        markers=True,
        title="Average Traffic Volume by Hour"
    )

    fig1.update_layout(
        template="plotly_white",
        paper_bgcolor="#fffbea",
        plot_bgcolor="#fffbea",
        height=450
    )

    st.plotly_chart(
        fig1,
        use_container_width=True
    )

    # =====================================================
    # TEMPERATURE + CLOUDS
    # =====================================================

    col1, col2 = st.columns(2)

    with col1:

        st.subheader(
            "🌡️ Temperature vs Traffic"
        )

        fig2 = px.scatter(
            df,
            x="temp",
            y="traffic_volume",
            title="Temperature vs Traffic",
            opacity=0.65
        )

        fig2.update_layout(
            template="plotly_white",
            paper_bgcolor="#fffbea",
            plot_bgcolor="#fffbea",
            height=400
        )

        st.plotly_chart(
            fig2,
            use_container_width=True
        )

    with col2:

        st.subheader(
            "☁️ Cloud Coverage vs Traffic"
        )

        fig3 = px.scatter(
            df,
            x="clouds_all",
            y="traffic_volume",
            title="Cloud Coverage vs Traffic",
            opacity=0.65
        )

        fig3.update_layout(
            template="plotly_white",
            paper_bgcolor="#fffbea",
            plot_bgcolor="#fffbea",
            height=400
        )

        st.plotly_chart(
            fig3,
            use_container_width=True
        )

    # =====================================================
    # WEATHER GRAPH
    # =====================================================

    st.header(
        "🌦️ Average Traffic by Weather"
    )

    weather_data = (
        df.groupby("weather_main")[
            "traffic_volume"
        ]
        .mean()
        .reset_index()
    )

    fig4 = px.bar(
        weather_data,
        x="weather_main",
        y="traffic_volume",
        title="Average Traffic by Weather"
    )

    fig4.update_layout(
        template="plotly_white",
        paper_bgcolor="#fffbea",
        plot_bgcolor="#fffbea",
        height=450
    )

    st.plotly_chart(
        fig4,
        use_container_width=True
    )

    # =====================================================
    # FEATURE IMPORTANCE
    # =====================================================

    st.header(
        "🌲 Random Forest Feature Importance"
    )

    try:

        feature_importance = pd.DataFrame({

            "Feature":
            feature_columns,

            "Importance":
            model.feature_importances_

        })

        feature_importance = (
            feature_importance
            .sort_values(
                "Importance",
                ascending=False
            )
        )

        top_features = (
            feature_importance
            .head(10)
        )

        fig5 = px.bar(
            top_features,
            x="Importance",
            y="Feature",
            orientation="h",
            title="Top 10 Important Features"
        )

        fig5.update_layout(
            template="plotly_white",
            paper_bgcolor="#fffbea",
            plot_bgcolor="#fffbea",
            height=500
        )

        st.plotly_chart(
            fig5,
            use_container_width=True
        )

        st.subheader(
            "📋 Feature Importance Table"
        )

        st.dataframe(
            feature_importance,
            use_container_width=True,
            hide_index=True
        )

    except Exception:

        st.info(
            "Feature importance is available for tree-based models."
        )


# =========================================================
# MODEL INFORMATION PAGE
# =========================================================

elif page == "🤖 Model Information":

    st.title(
        "🤖 Machine Learning Model"
    )

    st.write(
        "Information about the algorithms used in the project."
    )

    st.divider()

    # =====================================================
    # MODELS
    # =====================================================

    col1, col2, col3 = st.columns(3)

    with col1:

        st.subheader(
            "📈 Linear Regression"
        )

        st.write(
            """
            Predicts traffic volume by finding
            a linear relationship between the
            input features and traffic volume.
            """
        )

    with col2:

        st.subheader(
            "🌳 Decision Tree"
        )

        st.write(
            """
            Uses decision rules and tree branches
            to predict traffic volume.
            """
        )

    with col3:

        st.subheader(
            "🌲 Random Forest"
        )

        st.write(
            """
            Combines multiple decision trees
            to make the final prediction.
            """
        )

    st.divider()

    # =====================================================
    # DATASET INFORMATION
    # =====================================================

    st.header(
        "📁 Dataset Information"
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Total Rows",
            len(df)
        )

    with col2:

        st.metric(
            "Dataset Columns",
            len(df.columns)
        )

    with col3:

        st.metric(
            "Model Features",
            len(feature_columns)
        )

    st.divider()

    # =====================================================
    # MODEL FEATURES
    # =====================================================

    st.header(
        "🧩 Model Features"
    )

    feature_table = pd.DataFrame({

        "Feature":
        feature_columns

    })

    st.dataframe(
        feature_table,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    # =====================================================
    # PROJECT DESCRIPTION
    # =====================================================

    st.header(
        "ℹ️ About This Project"
    )

    st.info(
        """
        🚦 Traffic Volume Prediction

        This Machine Learning project predicts
        traffic volume using weather conditions
        and time-related features.

        The project uses:

        • Temperature
        • Rainfall
        • Snowfall
        • Cloud coverage
        • Weather conditions
        • Date and time information

        The Random Forest model is saved using
        Joblib and is used by this Streamlit
        web application.
        """
    )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "🚦 TrafficFlow AI | "
    "Machine Learning Traffic Volume Prediction System | "
    "Python • Pandas • Scikit-learn • Plotly • Streamlit"
)