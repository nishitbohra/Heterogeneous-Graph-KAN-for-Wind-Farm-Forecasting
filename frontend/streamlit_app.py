"""
Streamlit main application for Wind Power Forecasting.
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import io

# Add utils to path
sys.path.insert(0, str(Path(__file__).parent))

from utils.api_client import APIClient

# Page configuration
st.set_page_config(
    page_title="HG-KAN Wind Forecasting",
    page_icon="🌬️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .error-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'api_client' not in st.session_state:
    st.session_state.api_client = APIClient("http://localhost:8000")

if 'predictions' not in st.session_state:
    st.session_state.predictions = None

if 'input_data' not in st.session_state:
    st.session_state.input_data = None

if 'metrics' not in st.session_state:
    st.session_state.metrics = None


def main():
    """Main application."""
    
    # Header
    st.markdown('<div class="main-header">🌬️ Wind Power Forecasting</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Heterogeneous Graph-KAN for Multi-Horizon Prediction</div>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API Status
        st.subheader("🔌 API Status")
        health = st.session_state.api_client.health_check()
        
        if health.get('status') == 'healthy':
            st.success("✅ API Connected")
            st.info(f"Version: {health.get('version', 'N/A')}")
            
            # Model info
            model_info = st.session_state.api_client.get_model_info()
            if model_info:
                st.subheader("🤖 Model Info")
                st.write(f"**Type:** {model_info.get('model_type', 'N/A')}")
                st.write(f"**Nodes:** {model_info.get('num_nodes', 'N/A')}")
                st.write(f"**Input Window:** {model_info.get('input_window', 'N/A')}h")
                st.write(f"**Forecast Horizon:** {model_info.get('forecast_horizon', 'N/A')}h")
                st.write(f"**Parameters:** {model_info.get('parameters', 'N/A'):,}")
        else:
            st.error("❌ API Unavailable")
            st.warning("Please start the API server:\n```bash\npython -m api.main```")
            return
        
        st.divider()
        
        # Configuration options
        st.subheader("📊 Settings")
        forecast_horizon = st.slider("Forecast Horizon (hours)", 1, 6, 6)
        show_individual = st.checkbox("Show Individual Turbines", value=False)
        num_turbines_display = st.slider("Turbines to Display", 1, 20, 5) if show_individual else 5
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📤 Upload & Predict", "📈 Visualization", "🎯 Evaluation", "📚 Documentation"])
    
    with tab1:
        st.header("Upload Data & Generate Predictions")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Input Data")
            
            # Data input options
            input_method = st.radio(
                "Choose input method:",
                ["Upload CSV", "Upload NPY/NPZ", "Generate Sample Data"]
            )
            
            if input_method == "Upload CSV":
                uploaded_file = st.file_uploader(
                    "Upload CSV file with time series data",
                    type=['csv'],
                    help="CSV should have shape [timesteps, num_nodes] or [timesteps, features]"
                )
                
                if uploaded_file:
                    try:
                        df = pd.read_csv(uploaded_file)
                        st.success(f"✅ Loaded: {df.shape[0]} timesteps × {df.shape[1]} nodes")
                        
                        # Extract last N timesteps as input
                        model_info = st.session_state.api_client.get_model_info()
                        input_window = model_info.get('input_window', 24)
                        
                        if df.shape[0] < input_window:
                            st.error(f"❌ Need at least {input_window} timesteps")
                        else:
                            # Take last input_window timesteps
                            input_data = df.iloc[-input_window:].values.T  # [nodes, timesteps]
                            st.session_state.input_data = input_data
                            
                            st.info(f"Using last {input_window} timesteps for prediction")
                            st.write(f"Input shape: {input_data.shape} (nodes × timesteps)")
                            
                            # Show preview
                            with st.expander("Preview Data"):
                                st.dataframe(df.tail(input_window))
                    
                    except Exception as e:
                        st.error(f"❌ Error loading file: {str(e)}")
            
            elif input_method == "Upload NPY/NPZ":
                uploaded_file = st.file_uploader(
                    "Upload NPY or NPZ file",
                    type=['npy', 'npz'],
                    help="Array should have shape [num_nodes, input_window]"
                )
                
                if uploaded_file:
                    try:
                        if uploaded_file.name.endswith('.npy'):
                            input_data = np.load(uploaded_file)
                        else:
                            npz_file = np.load(uploaded_file)
                            # Try to find the right array
                            if 'input' in npz_file:
                                input_data = npz_file['input']
                            else:
                                input_data = npz_file[npz_file.files[0]]
                        
                        st.session_state.input_data = input_data
                        st.success(f"✅ Loaded: {input_data.shape}")
                        
                        with st.expander("Data Statistics"):
                            st.write(f"Shape: {input_data.shape}")
                            st.write(f"Mean: {input_data.mean():.4f}")
                            st.write(f"Std: {input_data.std():.4f}")
                            st.write(f"Min: {input_data.min():.4f}")
                            st.write(f"Max: {input_data.max():.4f}")
                    
                    except Exception as e:
                        st.error(f"❌ Error loading file: {str(e)}")
            
            else:  # Generate sample data
                st.info("📝 Generating synthetic sample data...")
                
                model_info = st.session_state.api_client.get_model_info()
                num_nodes = model_info.get('num_nodes', 200)
                input_window = model_info.get('input_window', 24)
                
                # Generate realistic wind power data (0-1 normalized)
                np.random.seed(42)
                t = np.linspace(0, 2*np.pi, input_window)
                base_pattern = 0.5 + 0.3 * np.sin(t)  # Diurnal pattern
                
                input_data = np.zeros((num_nodes, input_window))
                for i in range(num_nodes):
                    # Add node-specific variation
                    noise = np.random.randn(input_window) * 0.05
                    phase_shift = np.random.rand() * 2 * np.pi
                    input_data[i] = base_pattern + 0.1 * np.sin(t + phase_shift) + noise
                
                # Clip to [0, 1]
                input_data = np.clip(input_data, 0, 1)
                
                st.session_state.input_data = input_data
                st.success(f"✅ Generated: {num_nodes} turbines × {input_window} timesteps")
                
                with st.expander("View Sample Data"):
                    fig, ax = plt.subplots(figsize=(10, 4))
                    for i in range(min(5, num_nodes)):
                        ax.plot(input_data[i], alpha=0.7, label=f'Turbine {i+1}')
                    ax.set_xlabel('Time (hours)')
                    ax.set_ylabel('Normalized Power')
                    ax.set_title('Sample Input Data (First 5 Turbines)')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)
                    plt.close()
        
        with col2:
            st.subheader("Actions")
            
            predict_button = st.button(
                "🔮 Generate Prediction",
                type="primary",
                use_container_width=True,
                disabled=(st.session_state.input_data is None)
            )
            
            if predict_button and st.session_state.input_data is not None:
                with st.spinner("Generating predictions..."):
                    predictions, metadata, error = st.session_state.api_client.predict(
                        st.session_state.input_data,
                        forecast_horizon=forecast_horizon
                    )
                    
                    if error:
                        st.error(f"❌ Prediction failed: {error}")
                    else:
                        st.session_state.predictions = predictions
                        st.success("✅ Prediction completed!")
                        
                        # Show metadata
                        if metadata:
                            st.metric(
                                "Inference Time",
                                f"{metadata.get('inference_time_ms', 0):.1f} ms"
                            )
                        
                        st.info(f"Predicted {predictions.shape[0]} turbines for {predictions.shape[1]} hours ahead")
            
            st.divider()
            
            # Download buttons
            if st.session_state.predictions is not None:
                st.subheader("💾 Download Results")
                
                # Download as CSV
                csv_buffer = io.StringIO()
                pd.DataFrame(st.session_state.predictions).to_csv(csv_buffer, index=False)
                st.download_button(
                    "📄 Download CSV",
                    csv_buffer.getvalue(),
                    "predictions.csv",
                    "text/csv",
                    use_container_width=True
                )
                
                # Download as NPY
                npy_buffer = io.BytesIO()
                np.save(npy_buffer, st.session_state.predictions)
                st.download_button(
                    "📦 Download NPY",
                    npy_buffer.getvalue(),
                    "predictions.npy",
                    "application/octet-stream",
                    use_container_width=True
                )
    
    with tab2:
        st.header("Prediction Visualization")
        
        if st.session_state.predictions is None:
            st.info("ℹ️ No predictions available. Please generate predictions in the first tab.")
        else:
            predictions = st.session_state.predictions
            input_data = st.session_state.input_data
            
            # Summary statistics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Mean Prediction", f"{predictions.mean():.4f}")
            with col2:
                st.metric("Std Deviation", f"{predictions.std():.4f}")
            with col3:
                st.metric("Min Value", f"{predictions.min():.4f}")
            with col4:
                st.metric("Max Value", f"{predictions.max():.4f}")
            
            st.divider()
            
            # Visualizations
            if show_individual:
                st.subheader(f"Individual Turbine Predictions (First {num_turbines_display})")
                
                fig, axes = plt.subplots(
                    (num_turbines_display + 1) // 2, 2,
                    figsize=(14, 3 * ((num_turbines_display + 1) // 2))
                )
                axes = axes.flatten() if num_turbines_display > 1 else [axes]
                
                for i in range(min(num_turbines_display, predictions.shape[0])):
                    ax = axes[i]
                    
                    # Input history
                    input_len = input_data.shape[1]
                    ax.plot(
                        range(-input_len, 0),
                        input_data[i],
                        'o-',
                        label='History',
                        alpha=0.7
                    )
                    
                    # Predictions
                    ax.plot(
                        range(0, predictions.shape[1]),
                        predictions[i],
                        's-',
                        label='Forecast',
                        color='red',
                        alpha=0.7
                    )
                    
                    ax.axvline(0, color='black', linestyle='--', alpha=0.3)
                    ax.set_xlabel('Time (hours)')
                    ax.set_ylabel('Normalized Power')
                    ax.set_title(f'Turbine {i+1}')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                
                # Remove extra subplots
                for i in range(num_turbines_display, len(axes)):
                    fig.delaxes(axes[i])
                
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
            
            else:
                st.subheader("Aggregated Statistics")
                
                # Mean prediction across all turbines
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
                
                # Time series
                mean_input = input_data.mean(axis=0)
                mean_pred = predictions.mean(axis=0)
                std_pred = predictions.std(axis=0)
                
                input_len = input_data.shape[1]
                
                ax1.plot(range(-input_len, 0), mean_input, 'o-', label='Historical Mean', alpha=0.7)
                ax1.plot(range(0, len(mean_pred)), mean_pred, 's-', color='red', label='Forecast Mean', alpha=0.7)
                ax1.fill_between(
                    range(0, len(mean_pred)),
                    mean_pred - std_pred,
                    mean_pred + std_pred,
                    alpha=0.2,
                    color='red',
                    label='±1 Std'
                )
                ax1.axvline(0, color='black', linestyle='--', alpha=0.3)
                ax1.set_xlabel('Time (hours)')
                ax1.set_ylabel('Normalized Power')
                ax1.set_title('Mean Prediction Across All Turbines')
                ax1.legend()
                ax1.grid(True, alpha=0.3)
                
                # Distribution heatmap
                im = ax2.imshow(predictions[:20], aspect='auto', cmap='viridis', interpolation='nearest')
                ax2.set_xlabel('Forecast Horizon (hours)')
                ax2.set_ylabel('Turbine ID')
                ax2.set_title('Prediction Heatmap (First 20 Turbines)')
                plt.colorbar(im, ax=ax2, label='Normalized Power')
                
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
                
                # Distribution across forecast horizon
                st.subheader("Distribution Across Forecast Horizon")
                
                fig, ax = plt.subplots(figsize=(12, 5))
                positions = range(1, predictions.shape[1] + 1)
                bp = ax.boxplot(
                    [predictions[:, h] for h in range(predictions.shape[1])],
                    positions=positions,
                    widths=0.6,
                    patch_artist=True,
                    showmeans=True
                )
                
                for patch in bp['boxes']:
                    patch.set_facecolor('lightblue')
                
                ax.set_xlabel('Forecast Horizon (hours)')
                ax.set_ylabel('Normalized Power')
                ax.set_title('Prediction Distribution per Horizon')
                ax.grid(True, alpha=0.3, axis='y')
                
                st.pyplot(fig)
                plt.close()
    
    with tab3:
        st.header("Model Evaluation")
        
        st.info("📊 Upload ground truth data to evaluate predictions")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            ground_truth_file = st.file_uploader(
                "Upload ground truth data (CSV or NPY)",
                type=['csv', 'npy'],
                key='ground_truth'
            )
            
            if ground_truth_file and st.session_state.predictions is not None:
                try:
                    # Load ground truth
                    if ground_truth_file.name.endswith('.csv'):
                        gt_df = pd.read_csv(ground_truth_file)
                        ground_truth = gt_df.values
                    else:
                        ground_truth = np.load(ground_truth_file)
                    
                    # Validate shape
                    if ground_truth.shape != st.session_state.predictions.shape:
                        st.error(f"❌ Shape mismatch: predictions {st.session_state.predictions.shape} vs ground truth {ground_truth.shape}")
                    else:
                        st.success(f"✅ Ground truth loaded: {ground_truth.shape}")
                        
                        # Compute metrics
                        with st.spinner("Computing metrics..."):
                            metrics, error = st.session_state.api_client.evaluate(
                                st.session_state.predictions,
                                ground_truth
                            )
                            
                            if error:
                                st.error(f"❌ Evaluation failed: {error}")
                            else:
                                st.session_state.metrics = metrics
                                
                                # Display metrics
                                st.subheader("📈 Overall Metrics")
                                
                                col1, col2, col3, col4, col5 = st.columns(5)
                                col1.metric("MAE", f"{metrics['mae']:.4f}")
                                col2.metric("RMSE", f"{metrics['rmse']:.4f}")
                                col3.metric("MAPE", f"{metrics['mape']:.2f}%")
                                col4.metric("R²", f"{metrics['r2']:.4f}")
                                col5.metric("NRMSE", f"{metrics['nrmse']:.4f}")
                                
                                # Per-horizon metrics
                                if metrics.get('per_horizon'):
                                    st.subheader("📊 Metrics per Horizon")
                                    
                                    horizon_df = pd.DataFrame(metrics['per_horizon'])
                                    st.dataframe(horizon_df, use_container_width=True)
                                    
                                    # Plot metrics trend
                                    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
                                    
                                    axes[0].plot(horizon_df['horizon'], horizon_df['mae'], 'o-', label='MAE')
                                    axes[0].plot(horizon_df['horizon'], horizon_df['rmse'], 's-', label='RMSE')
                                    axes[0].set_xlabel('Forecast Horizon (hours)')
                                    axes[0].set_ylabel('Error')
                                    axes[0].set_title('Error Metrics vs Horizon')
                                    axes[0].legend()
                                    axes[0].grid(True, alpha=0.3)
                                    
                                    axes[1].plot(horizon_df['horizon'], horizon_df['r2'], 'o-', color='green')
                                    axes[1].set_xlabel('Forecast Horizon (hours)')
                                    axes[1].set_ylabel('R² Score')
                                    axes[1].set_title('R² Score vs Horizon')
                                    axes[1].grid(True, alpha=0.3)
                                    
                                    plt.tight_layout()
                                    st.pyplot(fig)
                                    plt.close()
                                
                                # Scatter plot
                                st.subheader("🎯 Prediction vs Ground Truth")
                                
                                fig, ax = plt.subplots(figsize=(8, 8))
                                
                                # Sample points for visualization
                                n_samples = min(5000, ground_truth.size)
                                indices = np.random.choice(ground_truth.size, n_samples, replace=False)
                                
                                gt_flat = ground_truth.flatten()[indices]
                                pred_flat = st.session_state.predictions.flatten()[indices]
                                
                                ax.scatter(gt_flat, pred_flat, alpha=0.3, s=10)
                                
                                # Perfect prediction line
                                min_val = min(gt_flat.min(), pred_flat.min())
                                max_val = max(gt_flat.max(), pred_flat.max())
                                ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
                                
                                ax.set_xlabel('Ground Truth')
                                ax.set_ylabel('Predictions')
                                ax.set_title('Prediction Scatter Plot')
                                ax.legend()
                                ax.grid(True, alpha=0.3)
                                
                                st.pyplot(fig)
                                plt.close()
                
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
        
        with col2:
            if st.session_state.metrics:
                st.subheader("💾 Export Metrics")
                
                # Download metrics as JSON
                import json
                metrics_json = json.dumps(st.session_state.metrics, indent=2)
                st.download_button(
                    "📄 Download Metrics (JSON)",
                    metrics_json,
                    "metrics.json",
                    "application/json",
                    use_container_width=True
                )
    
    with tab4:
        st.header("📚 Documentation")
        
        st.markdown("""
        ## About This Application
        
        This application provides a user-friendly interface for wind power forecasting using the 
        **Heterogeneous Graph-KAN (HG-KAN)** model, which combines:
        
        - 🧠 **Kolmogorov-Arnold Networks (KAN)**: Interpretable neural networks with learnable basis functions
        - 🌐 **Graph Neural Networks**: Capture spatial and temporal relationships
        - 🔗 **Heterogeneous Graphs**: Multiple edge types for different interaction patterns
        
        ### Model Architecture
        
        The model processes wind farm data through three types of edges:
        
        1. **Spatial Edges**: Based on geographic proximity (k-NN)
        2. **Wake Edges**: Directional effects based on wind direction
        3. **Correlation Edges**: Power output correlation patterns
        
        ### Input Requirements
        
        - **Format**: CSV, NPY, or NPZ file
        - **Shape**: `[num_nodes, input_window]` or `[timesteps, num_nodes]`
        - **Values**: Normalized power output (0-1 range recommended)
        - **Input Window**: Typically 24-48 hours of historical data
        
        ### Output Format
        
        - **Shape**: `[num_nodes, forecast_horizon]`
        - **Horizon**: 1-6 hours ahead (configurable)
        - **Values**: Predicted normalized power output
        
        ### How to Use
        
        1. **Start the API Server**:
           ```bash
           cd api
           python main.py
           ```
        
        2. **Launch Streamlit**:
           ```bash
           streamlit run frontend/streamlit_app.py
           ```
        
        3. **Upload Data**: Use the "Upload & Predict" tab
        4. **Generate Predictions**: Click the prediction button
        5. **Visualize Results**: Explore the visualization tab
        6. **Evaluate**: Upload ground truth for metric computation
        
        ### Performance Metrics
        
        - **MAE**: Mean Absolute Error
        - **RMSE**: Root Mean Squared Error
        - **MAPE**: Mean Absolute Percentage Error
        - **R²**: Coefficient of determination
        - **NRMSE**: Normalized RMSE
        
        ### Citation
        
        If you use this software in your research, please cite:
        
        ```bibtex
        @software{hg_kan_wind_forecasting,
          title={Heterogeneous Graph-KAN for Wind Farm Forecasting},
          author={Your Name},
          year={2024},
          url={https://github.com/yourusername/project}
        }
        ```
        
        ### Contact & Support
        
        - 📧 Email: your.email@domain.com
        - 🐛 Issues: GitHub Issues
        - 📖 Documentation: [Full Documentation](#)
        
        ---
        
        **Version**: 1.0.0  
        **Last Updated**: June 2026
        """)


if __name__ == "__main__":
    main()
