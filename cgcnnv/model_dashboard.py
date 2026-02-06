import streamlit as st
import torch
import numpy as np
import time
import plotly.express as px
import argparse

import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go

from torchviz import make_dot
import os

# Set Streamlit page config — this must come first!
st.set_page_config(page_title="Model Dashboard", layout="wide")

# Then comes the mode selection
mode = st.radio(
    "Select Dashboard Mode:",
    ["🎯 Built-in Model Analysis", "🧩 Load Custom Trained Model"],
    horizontal=True
)

BASE_PATH = os.path.abspath(os.path.dirname(__file__))

PREDICTIONS_FILE = os.path.join(BASE_PATH, "results", "predictions.csv")
TRAIN_RESULTS_FILE = os.path.join(BASE_PATH, "results", "train_results.csv")
VAL_RESULTS_FILE = os.path.join(BASE_PATH, "results", "val_results.csv")
TRAIN_LOSS_FILE = os.path.join(BASE_PATH, "results", "train_loss.csv")
train_loss_df = pd.read_csv(TRAIN_LOSS_FILE)
MODEL_FILE = os.path.join(BASE_PATH, "models", "model_best.pth.tar")

RESULTS2_DIR = os.path.join(BASE_PATH, "results2")
os.makedirs(RESULTS2_DIR, exist_ok=True)

from model import CrystalGraphConvNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def load_model():
    model = CrystalGraphConvNet(
        orig_atom_fea_len=92,
        nbr_fea_len=41,
        atom_fea_len=64,
        n_conv=3,
        h_fea_len=128,
        n_h=1,
        classification=False
    )
    return model

def force_zero_axis(fig, xmin=0, xmax=None, ymin=0, ymax=None):
    fig.update_layout(template="plotly_dark")

    if xmax is not None:
        fig.update_xaxes(range=[xmin, xmax])
    else:
        fig.update_xaxes(rangemode="tozero")

    if ymax is not None:
        fig.update_yaxes(range=[ymin, ymax])
    else:
        fig.update_yaxes(rangemode="tozero")

    return fig


def force_zero_axis(fig, xmin=0, xmax=None, ymin=0, ymax=None):
    fig.update_layout(
        template="plotly_dark",
        bargap=0.05
    )

    if xmax is not None:
        fig.update_xaxes(range=[xmin, xmax])
    else:
        fig.update_xaxes(rangemode="tozero")

    if ymax is not None:
        fig.update_yaxes(range=[ymin, ymax])
    else:
        fig.update_yaxes(rangemode="tozero")

    return fig




# Function to plot Predicted vs Actual values for comparison
def plot_pred_vs_actual(df1, df2=None, key_suffix=""):
    # Determine max sample size safely
    max_points = len(df1) if df2 is None else min(len(df1), len(df2))

    sample_size = st.slider(
        f"Select number of random points to display ({key_suffix or 'Model 1'})",
        min_value=100,
        max_value=max_points,
        value=min(3000, max_points),
        step=100,
        key=f"slider_{key_suffix}"
    )

    if df2 is not None:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Model 1")
            sample_df1 = df1.sample(n=min(sample_size, len(df1)), random_state=42)
            fig1 = go.Figure()
            fig1.add_trace(go.Scatter(
                x=sample_df1['Actual'],
                y=sample_df1['Predicted'],
                mode='markers',
                marker=dict(size=5, color='#FF6795', opacity=0.7),
                name='Model 1'
            ))
            ideal_min = min(sample_df1["Actual"].min(), sample_df1["Predicted"].min())
            ideal_max = max(sample_df1["Actual"].max(), sample_df1["Predicted"].max())
            fig1.add_trace(go.Scatter(
                x=[ideal_min, ideal_max],
                y=[ideal_min, ideal_max],
                mode='lines',
                line=dict(color='red', dash='dash'),
                name='Ideal'
            ))
            fig1.update_layout(
                title="",
                xaxis_title="Actual Band Gap",
                yaxis_title="Predicted Band Gap",
                template="plotly_dark",
                xaxis=dict(rangemode="tozero", range=[0, ideal_max]),
                yaxis=dict(rangemode="tozero", range=[0, ideal_max])
            )
            st.plotly_chart(fig1, use_container_width=True, key=f"plot_model1_{key_suffix}")

        with col2:
            st.subheader("Model 2")
            sample_df2 = df2.sample(n=min(sample_size, len(df2)), random_state=42)
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                x=sample_df2['Actual'],
                y=sample_df2['Predicted'],
                mode='markers',
                marker=dict(size=5, color='#0fba9c', opacity=0.7),
                name='Model 2'
            ))
            ideal_min = min(sample_df2["Actual"].min(), sample_df2["Predicted"].min())
            ideal_max = max(sample_df2["Actual"].max(), sample_df2["Predicted"].max())
            fig2.add_trace(go.Scatter(
                x=[ideal_min, ideal_max],
                y=[ideal_min, ideal_max],
                mode='lines',
                line=dict(color='red', dash='dash'),
                name='Ideal'
            ))
            fig2.update_layout(
                title="",
                xaxis_title="Actual Band Gap",
                yaxis_title="Predicted Band Gap",
                template="plotly_dark",
                xaxis=dict(rangemode="tozero", range=[0, ideal_max]),
                yaxis=dict(rangemode="tozero", range=[0, ideal_max])
            )
            st.plotly_chart(fig2, use_container_width=True, key=f"plot_model2_{key_suffix}")

    else:
        sample_df1 = df1.sample(n=min(sample_size, len(df1)), random_state=42)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=sample_df1['Actual'],
            y=sample_df1['Predicted'],
            mode='markers',
            marker=dict(size=5, color='#FF6795', opacity=0.7),
            name='Predictions'
        ))
        ideal_min = min(sample_df1["Actual"].min(), sample_df1["Predicted"].min())
        ideal_max = max(sample_df1["Actual"].max(), sample_df1["Predicted"].max())
        fig.add_trace(go.Scatter(
            x=[ideal_min, ideal_max],
            y=[ideal_min, ideal_max],
            mode='lines',
            line=dict(color='red', dash='dash'),
            name='Ideal'
        ))
        fig.update_layout(
            title="",
            xaxis_title="Actual Band Gap",
            yaxis_title="Predicted Band Gap",
            template="plotly_dark",
            xaxis=dict(rangemode="tozero", range=[0, ideal_max]),
            yaxis=dict(rangemode="tozero", range=[0, ideal_max])
        )
        st.plotly_chart(fig, use_container_width=True, key=f"plot_single_{key_suffix}")




# Function to calculate model metrics
def calculate_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    return {
        "MAE": mae,
        "RMSE": rmse,
        "R² Score": r2
    }



if mode == "🎯 Built-in Model Analysis":



    st.markdown(
        """
        <style>
            h1 {
                color: #0fba9c !important; /* Aixelo Green */
            }
        </style>
        """,
        unsafe_allow_html=True
    )

    LOGO_PATH = os.path.join(BASE_PATH, "assets", "logo dark.png")

    col1, col2 = st.columns([8, 1])

    with col1:
        st.title("📊 Model Dashboard")

    with col2:
        st.image(LOGO_PATH, width=50)

    # Load data
    df = pd.read_csv(PREDICTIONS_FILE, names=['CIF_ID', 'Actual', 'Predicted'])
    train_df = pd.read_csv(TRAIN_RESULTS_FILE)
    val_df = pd.read_csv(VAL_RESULTS_FILE)
    # Toggle for model comparison, placed in main area
    st.markdown("### 🔀 Model Comparison")
    st.markdown("""
        <style>
        .stToggle > label {
            font-size: 18px;
            font-weight: 600;
        }
        </style>
    """, unsafe_allow_html=True)
    compare_models = st.toggle("Compare with a second model", value=False)

    df1 = pd.read_csv(PREDICTIONS_FILE, names=['CIF_ID', 'Actual', 'Predicted'])
    df2 = None

    if compare_models:
        st.markdown("### 📥 Upload files for Model 2")
        st.markdown("Please upload all required files for the second model. The following files are expected:")
        st.markdown("- `predictions.csv`  \n- `train_results.csv`  \n- `val_results.csv`  \n- `train_loss.csv`")

        uploaded_files = st.file_uploader(
            "📁 Upload all Model 2 files",
            accept_multiple_files=True,
            type="csv"
        )

        for uploaded_file in uploaded_files:
            filename = uploaded_file.name
            save_path = os.path.join(RESULTS2_DIR, filename)
            with open(save_path, "wb") as f:
                f.write(uploaded_file.read())

        # Initialize df2 as None before loading
        df2 = None

        # Check if all required files are present
        expected_files = ["predictions.csv", "train_results.csv", "val_results.csv", "train_loss.csv"]
        missing_files = [f for f in expected_files if not os.path.exists(os.path.join(RESULTS2_DIR, f))]

        if missing_files:
            st.warning(f"⚠️ Please upload all required files for Model 2. Missing: {', '.join(missing_files)}")
        else:
            # Load only if all required files are available
            try:
                df2 = pd.read_csv(os.path.join(RESULTS2_DIR, "predictions.csv"), names=['CIF_ID', 'Actual', 'Predicted'])
            except Exception as e:
                st.error(f"❌ Failed to load Model 2 predictions: {e}")
                df2 = None



    # 📂 Dataset Overview
    st.header("📂 Dataset Overview")

    # Summary statistics
    st.subheader("📊 Summary Statistics")

    if compare_models:
        if df2 is not None:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### 📄 Model 1")
                st.markdown(f"""
                📝 **Number of records:** {df1.shape[0]}  
                📂 **Number of columns:** {df1.shape[1]}  
                🛑 **Missing values:** {df1.isnull().sum().sum()}  
                """)
                st.write(df1.describe())
            with col2:
                st.markdown("### 📄 Model 2")
                st.markdown(f"""
                📝 **Number of records:** {df2.shape[0]}  
                📂 **Number of columns:** {df2.shape[1]}  
                🛑 **Missing values:** {df2.isnull().sum().sum()}  
                """)
                st.write(df2.describe())
        else:
            st.info("📥 Please upload all required files for the second model to display comparison.")
    else:
        st.markdown(f"""
        📝 **Number of records:** {df1.shape[0]}  
        📂 **Number of columns:** {df1.shape[1]}  
        🛑 **Missing values:** {df1.isnull().sum().sum()}  
        """)
        st.write(df1.describe())


    # Distribution histograms
    st.subheader("📈 Distribution of Numerical Data")


    if compare_models:
        col1, col2 = st.columns(2)

        with col1:
            xmax1 = df1["Actual"].max() * 1.05
            ymax1 = int(df1["Actual"].value_counts(bins=50).max() * 1.2) + 500
            fig1 = px.histogram(
                df1, x="Actual", nbins=50,
                title="Distribution of Actual Band Gap Values (Model 1)",
                color_discrete_sequence=["#FF6795"]
            )
            st.plotly_chart(force_zero_axis(fig1, xmax=xmax1, ymax=ymax1), use_container_width=True)

            xmax2 = df1["Predicted"].max() * 1.05
            ymax2 = int(df1["Predicted"].value_counts(bins=50).max() * 1.2) + 500
            fig2 = px.histogram(
                df1, x="Predicted", nbins=50,
                title="Distribution of Predicted Band Gap Values (Model 1)",
                color_discrete_sequence=["#8DF6E3"]
            )
            st.plotly_chart(force_zero_axis(fig2, xmax=xmax2, ymax=ymax2), use_container_width=True)

        if df2 is not None:
            with col2:
                xmax3 = df2["Actual"].max() * 1.05
                ymax3 = int(df2["Actual"].value_counts(bins=50).max() * 1.2) + 500
                fig3 = px.histogram(
                    df2, x="Actual", nbins=50,
                    title="Distribution of Actual Band Gap Values (Model 2)",
                    color_discrete_sequence=["#80002A"]
                )
                st.plotly_chart(force_zero_axis(fig3, xmax=xmax3, ymax=ymax3), use_container_width=True)

                xmax4 = df2["Predicted"].max() * 1.05
                ymax4 = int(df2["Predicted"].value_counts(bins=50).max() * 1.2) + 500
                fig4 = px.histogram(
                    df2, x="Predicted", nbins=50,
                    title="Distribution of Predicted Band Gap Values (Model 2)",
                    color_discrete_sequence=["#0fba9c"]
                )
                st.plotly_chart(force_zero_axis(fig4, xmax=xmax4, ymax=ymax4), use_container_width=True)
        else:
            st.warning("📥 Please upload all required files for Model 2 to see comparison histograms.")

    else:
        xmax1 = df1["Actual"].max() * 1.05
        ymax1 = int(df1["Actual"].value_counts(bins=50).max() * 1.2) + 500
        fig1 = px.histogram(
            df1, x="Actual", nbins=50,
            title="Distribution of Actual Band Gap Values",
            color_discrete_sequence=["#FF6795"]
        )
        st.plotly_chart(force_zero_axis(fig1, xmax=xmax1, ymax=ymax1), use_container_width=True)

        xmax2 = df1["Predicted"].max() * 1.05
        ymax2 = int(df1["Predicted"].value_counts(bins=50).max() * 1.2) + 500
        fig2 = px.histogram(
            df1, x="Predicted", nbins=50,
            title="Distribution of Predicted Band Gap Values",
            color_discrete_sequence=["#8DF6E3"]
        )
        st.plotly_chart(force_zero_axis(fig2, xmax=xmax2, ymax=ymax2), use_container_width=True)


    # Custom CSS to enlarge tab labels
    st.markdown("""
        <style>
        .stTabs [data-baseweb="tab"] {
            font-size: 20px;
            padding: 1rem 2rem;
            margin-right: 10px;
            border-radius: 8px 8px 0 0;
            background-color: #1a1a1a;
            color: white;
            font-weight: 600;
        }
    
        .stTabs [aria-selected="true"] {
            background-color: #0fba9c;
            color: white;
            border-bottom: 3px solid #FF6795;
        }
        </style>
    """, unsafe_allow_html=True)

    # Create interactive tab navigation
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Predicted vs Actual",
        "📉 Training Loss",
        "📊 Model Metrics",
        "🧠 Model Architecture"
    ])

    # Tab 1: Predicted vs Actual
    with tab1:
        with st.container():
            st.header("📈 Predicted vs Actual Band Gap")

            if compare_models and df2 is not None:
                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("Model 1")
                    plot_pred_vs_actual(df1, key_suffix="model1")

                with col2:
                    st.subheader("Model 2")
                    plot_pred_vs_actual(df2, key_suffix="model2")
            else:
                plot_pred_vs_actual(df1, key_suffix="single")

            st.markdown("---")

    # Tab 2: Training Loss
    with tab2:
        with st.container():
            st.header("📉 Training Loss")

            if compare_models and df2 is not None:
                col1, col2 = st.columns(2)

                # Model 1
                with col1:
                    try:
                        train_loss_df1 = pd.read_csv(TRAIN_LOSS_FILE, header=0)
                        epochs1 = train_loss_df1["epoch"].astype(int).tolist()
                        losses1 = train_loss_df1["loss"].astype(float).tolist()

                        fig1 = go.Figure()
                        fig1.add_trace(go.Scatter(
                            x=epochs1,
                            y=losses1,
                            mode='lines+markers',
                            name='Model 1',
                            line=dict(color='#FF6795')
                        ))

                        fig1.update_layout(
                            title="First model Loss",
                            xaxis_title="Epoch",
                            yaxis_title="Loss",
                            template="plotly_dark",
                            yaxis=dict(range=[min(losses1) - 0.01, max(losses1) + 0.01])
                        )

                        st.plotly_chart(fig1, use_container_width=True)

                    except FileNotFoundError:
                        st.error("❌ train_loss.csv for Model 1 not found.")

                # Model 2
                with col2:
                    try:
                        train_loss_df2 = pd.read_csv(os.path.join(RESULTS2_DIR, "train_loss.csv"), header=0)
                        epochs2 = train_loss_df2["epoch"].astype(int).tolist()
                        losses2 = train_loss_df2["loss"].astype(float).tolist()

                        fig2 = go.Figure()
                        fig2.add_trace(go.Scatter(
                            x=epochs2,
                            y=losses2,
                            mode='lines+markers',
                            name='Model 2',
                            line=dict(color='#0fba9c')
                        ))

                        fig2.update_layout(
                            title="Second model loss",
                            xaxis_title="Epoch",
                            yaxis_title="Loss",
                            template="plotly_dark",
                            yaxis=dict(range=[min(losses2) - 0.01, max(losses2) + 0.01])
                        )

                        st.plotly_chart(fig2, use_container_width=True)

                    except FileNotFoundError:
                        st.error("❌ train_loss.csv for Model 2 not found.")
            else:
                try:
                    train_loss_df = pd.read_csv(TRAIN_LOSS_FILE, header=0)
                    epochs = train_loss_df["epoch"].astype(int).tolist()
                    losses = train_loss_df["loss"].astype(float).tolist()

                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=epochs,
                        y=losses,
                        mode='lines+markers',
                        name='Training Loss',
                        line=dict(color='#FF6795')
                    ))

                    fig.update_layout(
                        title="Model Loss Over Epochs",
                        xaxis_title="Epoch",
                        yaxis_title="Loss",
                        template="plotly_dark",
                        yaxis=dict(range=[min(losses) - 0.01, max(losses) + 0.01])
                    )

                    st.plotly_chart(fig, use_container_width=True)

                except FileNotFoundError:
                    st.error("❌ train_loss.csv was not found.")

            st.markdown("---")

    # Tab 3: Model Metrics
    with tab3:
        st.header("📊 Model Metrics")
        metrics1 = calculate_metrics(df1["Actual"], df1["Predicted"])

        if compare_models and df2 is not None:
            metrics2 = calculate_metrics(df2["Actual"], df2["Predicted"])
            col1, col2 = st.columns(2)
            with col1:
                st.metric("📏 MAE", f"{metrics1['MAE']:.3f}")
                st.metric("📏 RMSE", f"{metrics1['RMSE']:.3f}")
                st.metric("📊 R² Score", f"{metrics1['R² Score']:.3f}")
            with col2:
                st.metric("📏 MAE", f"{metrics2['MAE']:.3f}")
                st.metric("📏 RMSE", f"{metrics2['RMSE']:.3f}")
                st.metric("📊 R² Score", f"{metrics2['R² Score']:.3f}")
        elif compare_models:
            st.warning("📥 Please upload all required files for Model 2 to compare metrics.")
        else:
            st.metric("📏 MAE", f"{metrics1['MAE']:.3f}")
            st.metric("📏 RMSE", f"{metrics1['RMSE']:.3f}")
            st.metric("📊 R² Score", f"{metrics1['R² Score']:.3f}")


    # Tab 4: Model Architecture
    with tab4:
        st.header("🧠 Model Architecture")

        # Model 1
        model1 = load_model()
        total_params1 = sum(p.numel() for p in model1.parameters())
        trainable_params1 = sum(p.numel() for p in model1.parameters() if p.requires_grad)

        st.subheader("🧩 Model 1")
        with st.expander("🔍 Architecture", expanded=False):
            st.code(str(model1), language="python")
        st.markdown(f"- **Total Parameters:** `{total_params1}`")
        st.markdown(f"- **Trainable Parameters:** `{trainable_params1}`")

        # Model 2
        if compare_models and df2 is not None:
            model2 = load_model()
            total_params2 = sum(p.numel() for p in model2.parameters())
            trainable_params2 = sum(p.numel() for p in model2.parameters() if p.requires_grad)

            st.subheader("🧩 Model 2")
            with st.expander("🔍 Architecture", expanded=False):
                st.code(str(model2), language="python")
            st.markdown(f"- **Total Parameters:** `{total_params2}`")
            st.markdown(f"- **Trainable Parameters:** `{trainable_params2}`")
        elif compare_models:
            st.warning("📥 Please upload all required files for Model 2 to view architecture.")


else:
    import os
    import tempfile
    import traceback
    import tarfile
    import joblib
    import plotly.graph_objects as go
    import pandas as pd
    import streamlit as st
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    BASE_PATH = os.path.dirname(__file__)
    LOGO_PATH = os.path.join(BASE_PATH, "assets", "logo dark.png")



    model_type = st.selectbox("📌 Select Model Type", ["Fingerprint-based (CSV)", "CIF-based (CGCNN, PotNet)"])

    if model_type == "CIF-based (CGCNN, PotNet)":
        st.warning("🚧 CIF-based models (CGCNN, PotNet) are not yet supported in this dashboard.")
        uploaded_cif = st.file_uploader("Upload .cif or related files (not functional yet)", accept_multiple_files=True)
        st.stop()

    # ========================== CSV/PKL MODEL ==========================

    col1, col2 = st.columns([8, 1])
    with col1:
        st.markdown("## 🧩 Load and Analyze Custom Fingerprint Model")
    with col2:
        st.image(LOGO_PATH, width=5000)

    model_file = st.file_uploader("Upload your model checkpoint (.pth.tar, .pkl)", type=["tar", "pth", "pt"])
    features_file = st.file_uploader("Upload input features CSV", type="csv")
    target_file = st.file_uploader("Upload target values CSV (optional)", type="csv")
    loss_file = st.file_uploader("Upload train loss CSV (optional)", type="csv")

    if model_file and features_file:
        try:
            X = pd.read_csv(features_file)
            y = None
            if target_file:
                df_y = pd.read_csv(target_file)
                if df_y.shape[1] > 1:
                    target_col = st.selectbox("Select target column from uploaded CSV", df_y.columns)
                    y = df_y[target_col]
                else:
                    y = df_y.squeeze()

            X = X.select_dtypes(include=["number"])

            with tempfile.TemporaryDirectory() as tmpdir:
                with tarfile.open(fileobj=model_file, mode="r:*") as tar:
                    tar.extractall(path=tmpdir)
                model_path = os.path.join(tmpdir, "model.pkl")
                model = joblib.load(model_path)

                if hasattr(model, "feature_names_in_"):
                    model_features = list(model.feature_names_in_)
                    missing = set(model_features) - set(X.columns)
                    if missing:
                        st.warning(f"⚠️ Missing features required by the model: {missing}")
                    extra = set(X.columns) - set(model_features)
                    if extra:
                        st.info(f"ℹ️ Extra features in input (ignored): {extra}")
                    X = X[[col for col in model_features if col in X.columns]]

                y_pred = model.predict(X)
                pred_df = pd.DataFrame({"Predicted": y_pred})
                if y is not None:
                    pred_df["Actual"] = y.values

        except Exception as e:
            st.error("❌ Failed to load model or run prediction.")
            st.code(f"{type(e).__name__}: {e}", language="python")
            st.text("Stack trace:")
            st.text(traceback.format_exc())
            st.stop()

        tab1, tab2, tab3 = st.tabs(["📈 Predicted vs Actual", "📊 Model Metrics", "📉 Training Loss"])

        with tab1:
            st.subheader("Predicted vs Actual")
            if y is not None:
                sample_df = pred_df.sample(n=min(len(pred_df), 3000), random_state=42)
                try:
                    sample_df["Actual"] = sample_df["Actual"].astype(float)
                    min_val = 0  # Явно задаём нижнюю границу
                    max_val = max(sample_df["Actual"].max(), sample_df["Predicted"].max())

                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=sample_df["Actual"],
                        y=sample_df["Predicted"],
                        mode='markers',
                        marker=dict(size=5, color='#FF6795', opacity=0.7),
                        name='Predictions'
                    ))
                    fig.add_trace(go.Scatter(
                        x=[min_val, max_val],
                        y=[min_val, max_val],
                        mode='lines',
                        line=dict(color='red', dash='dash'),
                        name='Ideal'
                    ))
                    fig.update_layout(
                        xaxis_title="Actual",
                        yaxis_title="Predicted",
                        xaxis=dict(range=[0, max_val]),
                        yaxis=dict(range=[0, max_val]),
                        template="plotly_dark"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error("⚠️ Error rendering scatterplot.")
                    st.text(traceback.format_exc())
            else:
                st.info("No target CSV provided. Only predictions are shown.")

        with tab2:
            st.subheader("Model Metrics")
            if y is not None:
                try:
                    mae = mean_absolute_error(y, y_pred)
                    rmse = mean_squared_error(y, y_pred) ** 0.5
                    r2 = r2_score(y, y_pred)
                    st.metric("MAE", f"{mae:.3f}")
                    st.metric("RMSE", f"{rmse:.3f}")
                    st.metric("R² Score", f"{r2:.3f}")
                except Exception as e:
                    st.error("⚠️ Error computing metrics.")
                    st.code(f"{type(e).__name__}: {e}", language="python")
                    st.text(traceback.format_exc())
            else:
                st.info("Target not provided — metrics not available.")

        with tab3:
            st.subheader("Training Loss")
            if loss_file:
                try:
                    loss_df = pd.read_csv(loss_file)
                    if "epoch" not in loss_df.columns or "loss" not in loss_df.columns:
                        st.warning("⚠️ Loss file must contain 'epoch' and 'loss' columns.")
                    else:
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=loss_df["epoch"],
                            y=loss_df["loss"],
                            mode="lines+markers",
                            name="Training Loss",
                            line=dict(color="#0fba9c")
                        ))
                        fig.update_layout(
                            xaxis_title="Epoch",
                            yaxis_title="Loss",
                            template="plotly_dark"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error("⚠️ Failed to read loss file.")
                    st.text(traceback.format_exc())
            else:
                st.info("No training loss file uploaded.")
    else:
        st.info("Pleasе upload both a model checkpoint and a features CSV to proceed.")


















