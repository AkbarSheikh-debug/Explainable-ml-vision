import streamlit as st
import os
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from streamlit_option_menu import option_menu
import pandas as pd
import io
import sys
from skimage.segmentation import mark_boundaries

# Add current directory to path
sys.path.append('.')

from utils import ExplainableML
import train

# Page configuration
st.set_page_config(
    page_title="Explainable ML Vision",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #3B82F6;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .info-box {
        background-color: #F0F9FF;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #3B82F6;
        margin: 1rem 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 1rem;
        border: 2px solid #E2E8F0;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: transform 0.2s;
        color: white;
        margin: 0.5rem 0;
    }
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
    }
    .metric-card h3 {
        font-size: 2.5rem;
        margin: 0.5rem 0;
    }
    .metric-card h4 {
        font-size: 1rem;
        margin: 0.5rem 0;
        opacity: 0.9;
        font-weight: 500;
    }
    .metric-card h2 {
        font-size: 2rem;
        margin: 0.5rem 0;
        font-weight: bold;
    }
    .prediction-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 1rem;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        color: white;
        margin: 1rem 0;
    }
    .prediction-card h4 {
        font-size: 1.2rem;
        margin-bottom: 1rem;
        opacity: 0.9;
        font-weight: 500;
    }
    .prediction-card h2 {
        font-size: 2.5rem;
        margin: 0.5rem 0;
        font-weight: bold;
    }
    .confidence-high {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
    }
    .confidence-medium {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    }
    .confidence-low {
        background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
    }
    .stats-container {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1.5rem;
        border-radius: 1rem;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    .prediction-container {
        background: linear-gradient(135deg, #e0c3fc 0%, #8ec5fc 100%);
        padding: 2rem;
        border-radius: 1rem;
        margin: 1rem 0;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
</style>
""", unsafe_allow_html=True)

class ExplainableMLApp:
    def __init__(self):
        self.explainer = None
        self.model = None
        self.class_names = ['ants', 'bees']
        self.model_loaded = False
        
    def load_model(self, model_path):
        """Load a trained model"""
        try:
            if not os.path.exists(model_path):
                st.error(f"Model file not found at {model_path}")
                return False

            # Load checkpoint. Newer PyTorch versions default to `weights_only=True`
            # which restricts globals allowed during unpickling. Allowlist
            # numpy's internal reconstruct function so checkpoints saved with
            # numpy objects can be loaded safely.
            try:
                with torch.serialization.safe_globals([np._core.multiarray._reconstruct]):
                    checkpoint = torch.load(model_path, map_location=torch.device('cpu'), weights_only=True)
            except Exception:
                # Fallback to full load if the safe-globals approach fails
                checkpoint = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
            
            # Initialize model
            model_name = checkpoint.get('model_name', 'resnext101')
            model, input_size = self.explainer.initialize_model(model_name=model_name)
            
            # Load weights
            model.load_state_dict(checkpoint['model_state_dict'])
            model = model.to(self.explainer.device)
            model.eval()
            
            self.model = model
            self.class_names = checkpoint.get('class_names', self.class_names)
            self.model_loaded = True
            
            return True
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            return False
    
    def run(self):
        """Main application"""
        st.markdown('<h1 class="main-header">🔍 Explainable ML Vision</h1>', unsafe_allow_html=True)
        
        # Initialize explainer
        if self.explainer is None:
            self.explainer = ExplainableML()
        
        # Sidebar navigation
        with st.sidebar:
            st.image("https://img.icons8.com/color/96/000000/artificial-intelligence.png", width=100)
            st.markdown("### Navigation")
            
            selected = option_menu(
                menu_title=None,
                options=["Home", "Model Training", "Image Analysis", "EDA & Metrics", "About"],
                icons=["house", "gear", "image", "bar-chart", "info-circle"],
                default_index=0,
            )
        
        # Home Page
        if selected == "Home":
            self.home_page()
        
        # Model Training Page
        elif selected == "Model Training":
            self.training_page()
        
        # Image Analysis Page
        elif selected == "Image Analysis":
            self.analysis_page()
        
        # EDA & Metrics Page
        elif selected == "EDA & Metrics":
            self.metrics_page()
        
        # About Page
        elif selected == "About":
            self.about_page()
    
    def home_page(self):
        """Home page content"""
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            ### Welcome to Explainable ML Vision
            
            This application provides a comprehensive platform for:
            
            - **Training** deep learning models on custom image datasets
            - **Analyzing** predictions with multiple explainability techniques
            - **Visualizing** model behavior and decision-making process
            
            ### Key Features:
            
            🔧 **Model Training**: Train state-of-the-art models on your image dataset
            
            📊 **Explainability**: 
            - **Grad-CAM**: Visualize important regions for predictions
            - **LIME**: Local interpretable model-agnostic explanations
            - **SHAP**: SHapley Additive exPlanations
            
            📈 **Visualization**:
            - Training history plots
            - Confusion matrices
            - Feature importance
            
            ### Get Started:
            
            1. Upload your dataset in the proper folder structure
            2. Train a model in the **Model Training** section
            3. Analyze images in the **Image Analysis** section
            4. View metrics in the **EDA & Metrics** section
            """)
        
        with col2:
            st.markdown('<h3 style="color: #1E3A8A; font-size: 1.5rem; margin-bottom: 1rem;">📊 Quick Stats</h3>', unsafe_allow_html=True)
            
            # Check if model exists
            model_exists = os.path.exists("models/resnext101_best.pth")
            
            # Use Streamlit metrics for better visibility
            col_stat1, col_stat2 = st.columns(2)
            
            with col_stat1:
                status_icon = "✅" if model_exists else "❌"
                status_text = "Ready" if model_exists else "Not Ready"
                status_color = "normal" if model_exists else "off"
                st.metric(
                    label="🎯 Model Status",
                    value=status_text,
                    delta=status_icon,
                    delta_color=status_color
                )
            
            with col_stat2:
                st.metric(
                    label="📁 Classes",
                    value="2",
                    delta="ants, bees"
                )
            
            # Additional stats with enhanced cards
            st.markdown('<div class="stats-container">', unsafe_allow_html=True)
            st.markdown("### 📈 Model Information")
            
            if model_exists:
                try:
                    # Try to load model info
                    with torch.serialization.safe_globals([np._core.multiarray._reconstruct]):
                        checkpoint = torch.load("models/resnext101_best.pth", map_location=torch.device('cpu'), weights_only=True)
                except Exception:
                    try:
                        checkpoint = torch.load("models/resnext101_best.pth", map_location=torch.device('cpu'), weights_only=False)
                    except Exception:
                        checkpoint = {}
                
                model_name = checkpoint.get('model_name', 'resnext101')
                num_epochs = len(checkpoint.get('train_losses', []))
                final_acc = checkpoint.get('val_accs', [0])[-1] if checkpoint.get('val_accs') else 0
                
                col_info1, col_info2 = st.columns(2)
                with col_info1:
                    st.markdown(f"**Model:** {model_name}")
                    st.markdown(f"**Epochs Trained:** {num_epochs}")
                with col_info2:
                    st.markdown(f"**Best Val Acc:** {final_acc:.1%}" if final_acc > 0 else "**Status:** Trained")
            else:
                st.info("Train a model to see detailed statistics")
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown("### Dataset Structure")
            st.code("""
            hymenoptera_data/
            ├── train/
            │   ├── ants/
            │   └── bees/
            ├── val/
            │   ├── ants/
            │   └── bees/
            └── test/
                ├── ants/
                └── bees/
            """)
    
    def training_page(self):
        """Model training page"""
        st.markdown('<h2 class="sub-header">🛠️ Model Training</h2>', unsafe_allow_html=True)
        
        # Training parameters with expandable sections
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("### Training Configuration")
            
            # Basic Model Settings
            with st.expander("📐 Model & Basic Settings", expanded=True):
                col_model1, col_model2 = st.columns(2)
                
                with col_model1:
                    model_choice = st.selectbox(
                        "Model Architecture",
                        ["resnext101", "resnet50", "efficientnet_b0"],
                        index=0,
                        help="Choose the base model architecture"
                    )
                    
                    epochs = st.slider("Number of Epochs", 5, 200, 25, help="Total number of training epochs")
                    batch_size = st.slider("Batch Size", 4, 128, 32, step=4, help="Number of samples per batch")
                
                with col_model2:
                    learning_rate = st.number_input("Learning Rate", 0.00001, 0.1, 0.001, step=0.0001, format="%.5f", help="Initial learning rate")
                    
                    feature_extract = st.checkbox("Feature Extraction Mode", value=True, help="Freeze pretrained weights (faster training)")
                    
                    num_workers = st.slider("Data Loader Workers", 0, 8, 4, help="Number of parallel workers for data loading (0 = main thread only)")
            
            # Optimizer Settings
            with st.expander("⚙️ Optimizer Settings"):
                col_opt1, col_opt2 = st.columns(2)
                
                with col_opt1:
                    optimizer_choice = st.selectbox(
                        "Optimizer",
                        ["adam", "adamw", "sgd"],
                        index=0,
                        help="Optimization algorithm"
                    )
                    
                    weight_decay = st.number_input("Weight Decay (L2 Regularization)", 0.0, 0.01, 0.0001, step=0.0001, format="%.5f", help="L2 regularization coefficient")
                
                with col_opt2:
                    momentum = st.number_input("Momentum (for SGD)", 0.0, 0.99, 0.9, step=0.01, help="Momentum factor (only used with SGD)")
                    st.caption("💡 Momentum is only used when SGD optimizer is selected")
            
            # Learning Rate Scheduler Settings
            with st.expander("📉 Learning Rate Scheduler"):
                col_sched1, col_sched2 = st.columns(2)
                
                with col_sched1:
                    scheduler_choice = st.selectbox(
                        "Scheduler Type",
                        ["steplr", "cosine", "exponential"],
                        index=0,
                        help="Learning rate scheduling strategy"
                    )
                    
                    scheduler_step_size = st.number_input("Step Size (for StepLR)", 1, 50, 7, help="Period of learning rate decay")
                
                with col_sched2:
                    scheduler_gamma = st.number_input("Gamma (Decay Factor)", 0.01, 1.0, 0.1, step=0.01, help="Multiplicative factor for learning rate decay")
                    st.caption("💡 Step Size and Gamma are used for StepLR and ExponentialLR schedulers")
        
        with col2:
            st.markdown("### Dataset Information")
            
            # Check dataset structure
            data_dir = "data/hymenoptera_data"
            
            if os.path.exists(data_dir):
                train_ants = len(os.listdir(os.path.join(data_dir, "train", "ants"))) if os.path.exists(os.path.join(data_dir, "train", "ants")) else 0
                train_bees = len(os.listdir(os.path.join(data_dir, "train", "bees"))) if os.path.exists(os.path.join(data_dir, "train", "bees")) else 0
                val_ants = len(os.listdir(os.path.join(data_dir, "val", "ants"))) if os.path.exists(os.path.join(data_dir, "val", "ants")) else 0
                val_bees = len(os.listdir(os.path.join(data_dir, "val", "bees"))) if os.path.exists(os.path.join(data_dir, "val", "bees")) else 0
                
                st.info(f"""
                **Dataset Structure Found:**
                - Train Ants: {train_ants} images
                - Train Bees: {train_bees} images
                - Val Ants: {val_ants} images
                - Val Bees: {val_bees} images
                
                **Total:** {train_ants + train_bees + val_ants + val_bees} images
                """)
            else:
                st.warning("Dataset directory not found. Please ensure data is in 'data/hymenoptera_data/'")
            
            # Training Summary
            st.markdown("### Training Summary")
            st.markdown(f"""
            **Configuration Preview:**
            - Model: {model_choice}
            - Epochs: {epochs}
            - Batch Size: {batch_size}
            - Learning Rate: {learning_rate:.5f}
            - Optimizer: {optimizer_choice.upper()}
            - Scheduler: {scheduler_choice}
            - Feature Extract: {'Yes' if feature_extract else 'No'}
            """)
        
        # Start training button
        if st.button("🚀 Start Training", type="primary", width='stretch'):
            if not os.path.exists(data_dir):
                st.error("Dataset directory not found!")
                return
            
            with st.spinner("Training in progress..."):
                try:
                    # Create progress bar
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Train model (simulated progress for demo)
                    for i in range(100):
                        progress_bar.progress(i + 1)
                        status_text.text(f"Training... {i+1}%")
                        # In real implementation, this would call the actual training function
                    
                    # Call actual training function with all parameters
                    model_path, train_losses, val_losses, train_accs, val_accs = train.train_model(
                        data_dir=data_dir,
                        model_name=model_choice,
                        num_epochs=epochs,
                        batch_size=batch_size,
                        learning_rate=learning_rate,
                        optimizer_name=optimizer_choice,
                        weight_decay=weight_decay,
                        momentum=momentum,
                        scheduler_type=scheduler_choice,
                        scheduler_step_size=scheduler_step_size,
                        scheduler_gamma=scheduler_gamma,
                        feature_extract=feature_extract,
                        num_workers=num_workers
                    )
                    
                    # Display training results
                    st.success(f"✅ Training completed! Model saved to {model_path}")
                    
                    # Plot training history
                    fig, buf = self.explainer.plot_training_history(
                        train_losses, val_losses, train_accs, val_accs
                    )
                    st.pyplot(fig)
                    
                    # Display metrics
                    final_train_acc = train_accs[-1] if train_accs else 0
                    final_val_acc = val_accs[-1] if val_accs else 0
                    
                    col_met1, col_met2, col_met3 = st.columns(3)
                    
                    with col_met1:
                        st.metric("Final Training Accuracy", f"{final_train_acc:.2%}")
                    
                    with col_met2:
                        st.metric("Final Validation Accuracy", f"{final_val_acc:.2%}")
                    
                    with col_met3:
                        st.metric("Total Epochs", epochs)
                    
                except Exception as e:
                    st.error(f"Training failed: {str(e)}")
    
    def analysis_page(self):
        """Image analysis page"""
        st.markdown('<h2 class="sub-header">🔍 Image Analysis & Explainability</h2>', unsafe_allow_html=True)
        
        # Check if model is loaded
        model_path = "models/resnext101_best.pth"
        
        if not os.path.exists(model_path):
            st.warning("⚠️ No trained model found. Please train a model first.")
            return
        
        # Load model if not already loaded
        if not self.model_loaded:
            with st.spinner("Loading model..."):
                if self.load_model(model_path):
                    st.success("✅ Model loaded successfully!")
                else:
                    st.error("Failed to load model")
                    return
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Upload an image for analysis",
            type=['png', 'jpg', 'jpeg', 'bmp']
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### Original Image")
                image = Image.open(uploaded_file)
                st.image(image, width='stretch')
                
                # Save uploaded file temporarily
                temp_path = "temp_uploaded_image.jpg"
                image.save(temp_path)
            
            with col2:
                # Get prediction
                with st.spinner("Analyzing image..."):
                    try:
                        # Get target layer for Grad-CAM
                        target_layer = self.explainer.get_target_layer(self.model)
                        
                        # Generate Grad-CAM
                        cam_result, pred_class, probs = self.explainer.generate_grad_cam(
                            temp_path, 
                            self.model,
                            target_layer
                        )
                        
                        # Display prediction
                        predicted_class = self.class_names[pred_class]
                        confidence = probs[pred_class]
                        
                        # Determine confidence level for styling
                        if confidence >= 0.8:
                            conf_class = "confidence-high"
                            conf_label = "High Confidence"
                        elif confidence >= 0.5:
                            conf_class = "confidence-medium"
                            conf_label = "Medium Confidence"
                        else:
                            conf_class = "confidence-low"
                            conf_label = "Low Confidence"
                        
                        st.markdown('<div class="prediction-container">', unsafe_allow_html=True)
                        st.markdown('<h2 style="color: #1E3A8A; font-size: 2rem; text-align: center; margin-bottom: 1.5rem;">🎯 Prediction Results</h2>', unsafe_allow_html=True)
                        
                        col_pred1, col_pred2 = st.columns(2)
                        
                        with col_pred1:
                            st.markdown(f"""
                            <div class="prediction-card">
                                <h4>Predicted Class</h4>
                                <h2 style="font-size: 3rem;">{predicted_class.upper()}</h2>
                                <p style="font-size: 1.2rem; margin-top: 1rem;">🎯 Classification Result</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col_pred2:
                            st.markdown(f"""
                            <div class="prediction-card {conf_class}">
                                <h4>Confidence Score</h4>
                                <h2 style="font-size: 3rem;">{confidence:.1%}</h2>
                                <p style="font-size: 1.2rem; margin-top: 1rem;">{conf_label}</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Add progress bar for confidence visualization
                        st.markdown("### Confidence Level")
                        st.progress(float(confidence))
                        st.caption(f"Model is {confidence:.1%} confident this image is a **{predicted_class}**")
                        
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Display class probabilities with enhanced visualization
                        st.markdown("### 📊 Class Probabilities Distribution")
                        prob_df = pd.DataFrame({
                            'Class': self.class_names,
                            'Probability': probs
                        })
                        
                        # Create a more interactive and colorful chart
                        fig_prob = px.bar(
                            prob_df, 
                            x='Class', 
                            y='Probability',
                            color='Probability',
                            color_continuous_scale='Viridis',
                            title='Prediction Probabilities for Each Class',
                            labels={'Probability': 'Probability (%)', 'Class': 'Class Name'},
                            text=[f'{p:.1%}' for p in probs]
                        )
                        fig_prob.update_traces(
                            texttemplate='%{text}',
                            textposition='outside',
                            marker=dict(line=dict(color='#000000', width=2)),
                            hovertemplate='<b>%{x}</b><br>Probability: %{y:.2%}<extra></extra>'
                        )
                        fig_prob.update_layout(
                            height=400,
                            showlegend=False,
                            font=dict(size=14),
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)'
                        )
                        st.plotly_chart(fig_prob, width='stretch', use_container_width=True)
                        
                        # Add probability cards
                        st.markdown("### 📈 Detailed Probabilities")
                        prob_col1, prob_col2 = st.columns(2)
                        
                        for idx, (class_name, prob) in enumerate(zip(self.class_names, probs)):
                            with prob_col1 if idx == 0 else prob_col2:
                                # Determine color based on probability
                                if prob == max(probs):
                                    card_color = "linear-gradient(135deg, #11998e 0%, #38ef7d 100%)"
                                    badge = "🏆 Winner"
                                else:
                                    card_color = "linear-gradient(135deg, #667eea 0%, #764ba2 100%)"
                                    badge = ""
                                
                                st.markdown(f"""
                                <div style="background: {card_color}; padding: 1.5rem; border-radius: 1rem; 
                                            text-align: center; color: white; margin: 0.5rem 0; 
                                            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);">
                                    <h3 style="font-size: 1.5rem; margin: 0.5rem 0;">{class_name.upper()}</h3>
                                    <h2 style="font-size: 2.5rem; margin: 0.5rem 0; font-weight: bold;">{prob:.1%}</h2>
                                    <p style="font-size: 1rem; margin-top: 0.5rem;">{badge}</p>
                                </div>
                                """, unsafe_allow_html=True)
                        
                    except Exception as e:
                        st.error(f"Error during analysis: {str(e)}")
            
            # Explanations section
            st.markdown("---")
            st.markdown('<h3 class="sub-header">📊 Explanations</h3>', unsafe_allow_html=True)
            
            # Create tabs for different explanation methods
            tab1, tab2, tab3 = st.tabs(["Grad-CAM", "LIME", "SHAP"])
            
            with tab1:
                st.markdown("### Grad-CAM Visualization")
                if 'cam_result' in locals():
                    col_cam1, col_cam2 = st.columns(2)
                    
                    with col_cam1:
                        st.markdown("**Heatmap Overlay**")
                        st.image(cam_result, width='stretch')
                    
                    with col_cam2:
                        st.markdown("**How to interpret:**")
                        st.markdown("""
                        - **Red/Orange areas**: Most important regions for prediction
                        - **Blue areas**: Less important regions
                        - The model focuses on discriminative features
                        - Helps understand what the model "sees"
                        """)
            
            with tab2:
                st.markdown("### LIME Explanation")
                if st.button("Generate LIME Explanation", key="lime_btn"):
                    with st.spinner("Generating LIME explanation..."):
                        try:
                            lime_explanation, img_array = self.explainer.generate_lime_explanation(
                                temp_path, self.model
                            )
                            
                            # Display LIME explanation
                            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                            
                            # Original image
                            axes[0].imshow(img_array)
                            axes[0].set_title("Original Image")
                            axes[0].axis('off')
                            
                            # LIME explanation for top class
                            temp, mask = lime_explanation.get_image_and_mask(
                                lime_explanation.top_labels[0],
                                positive_only=True,
                                num_features=10,
                                hide_rest=False
                            )
                            axes[1].imshow(mark_boundaries(temp / 255.0, mask))
                            axes[1].set_title("LIME Explanation")
                            axes[1].axis('off')
                            
                            # Superpixels
                            axes[2].imshow(lime_explanation.segments)
                            axes[2].set_title("Superpixels")
                            axes[2].axis('off')
                            
                            plt.tight_layout()
                            st.pyplot(fig)
                            
                        except Exception as e:
                            st.error(f"Error generating LIME: {str(e)}")
            
            with tab3:
                st.markdown("### SHAP Explanation")
                if st.button("Generate SHAP Explanation", key="shap_btn"):
                    with st.spinner("Generating SHAP explanation (this may take a minute)..."):
                        try:
                            shap_numpy, test_numpy = self.explainer.generate_shap_explanation(
                                temp_path, self.model
                            )
                            
                            # Validate SHAP output
                            if shap_numpy is None or len(shap_numpy) == 0:
                                st.error("SHAP explanation returned empty results. Please try again.")
                                return
                            
                            # Display SHAP explanation
                            fig_shap, axes_shap = plt.subplots(1, min(3, len(shap_numpy) + 1), figsize=(15, 5))
                            
                            # Ensure axes_shap is iterable even if single subplot
                            if len(shap_numpy) == 1:
                                axes_shap = [axes_shap]
                            
                            # Original image
                            if test_numpy is not None and len(test_numpy) > 0:
                                # Normalize test image for display
                                test_img = test_numpy[0]
                                if test_img.max() > 1.0:
                                    test_img = test_img / 255.0
                                # Denormalize for display (reverse ImageNet normalization)
                                test_img = np.clip(test_img, 0, 1)
                                axes_shap[0].imshow(test_img)
                            else:
                                # Fallback to original uploaded image
                                image = Image.open(temp_path)
                                axes_shap[0].imshow(image)
                            axes_shap[0].set_title("Original Image")
                            axes_shap[0].axis('off')
                            
                            # Display SHAP values for each class
                            for idx, shap_class in enumerate(shap_numpy):
                                if idx + 1 >= len(axes_shap):
                                    break
                                
                                # Handle different shapes
                                if isinstance(shap_class, np.ndarray):
                                    shap_img = shap_class
                                    # Handle different array shapes
                                    if len(shap_img.shape) == 4:
                                        shap_img = shap_img[0]  # Take first batch
                                    if len(shap_img.shape) == 3:
                                        # If shape is (H, W, C), take mean across channels or first channel
                                        if shap_img.shape[2] == 3:
                                            shap_img = shap_img.mean(axis=2)  # Average across RGB
                                        else:
                                            shap_img = shap_img[:, :, 0]
                                    
                                    # Normalize SHAP values for visualization
                                    shap_min = shap_img.min()
                                    shap_max = shap_img.max()
                                    if shap_max - shap_min > 0:
                                        shap_img = (shap_img - shap_min) / (shap_max - shap_min)
                                    
                                    # Use a colormap that shows positive (red) and negative (blue) values
                                    axes_shap[idx + 1].imshow(shap_img, cmap='RdBu_r', vmin=0, vmax=1)
                                    axes_shap[idx + 1].set_title(f"SHAP for {self.class_names[idx] if idx < len(self.class_names) else f'Class {idx}'}")
                                else:
                                    st.warning(f"Unexpected SHAP value format for class {idx}")
                                
                                axes_shap[idx + 1].axis('off')
                            
                            plt.tight_layout()
                            st.pyplot(fig_shap)
                            
                            st.markdown("""
                            **SHAP Interpretation:**
                            - **Red areas**: Increase probability of the class
                            - **Blue areas**: Decrease probability of the class
                            - Shows how each pixel contributes to the prediction
                            """)
                            
                        except Exception as e:
                            import traceback
                            error_details = traceback.format_exc()
                            st.error(f"Error generating SHAP: {str(e)}")
                            st.code(error_details, language='python')
    
    def metrics_page(self):
        """EDA and Metrics page"""
        st.markdown('<h2 class="sub-header">📈 EDA & Model Metrics</h2>', unsafe_allow_html=True)
        
        # Check if model exists
        model_path = "models/resnext101_best.pth"
        
        if not os.path.exists(model_path):
            st.warning("No trained model found. Train a model first to see metrics.")
            return
        
        # Load model checkpoint with safe globals for numpy objects
        try:
            with torch.serialization.safe_globals([np._core.multiarray._reconstruct]):
                checkpoint = torch.load(model_path, map_location=torch.device('cpu'), weights_only=True)
        except Exception:
            # Fallback to full load if the safe-globals approach fails
            checkpoint = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
        
        # Display training history
        if 'train_losses' in checkpoint and 'val_losses' in checkpoint:
            col_hist1, col_hist2 = st.columns(2)
            
            with col_hist1:
                st.markdown("### Training History")
                
                # Create dataframe for plotting
                epochs = range(1, len(checkpoint['train_losses']) + 1)
                history_df = pd.DataFrame({
                    'Epoch': list(epochs) * 2,
                    'Loss': checkpoint['train_losses'] + checkpoint['val_losses'],
                    'Type': ['Train'] * len(epochs) + ['Validation'] * len(epochs)
                })
                
                fig_loss = px.line(history_df, x='Epoch', y='Loss', color='Type',
                                 title='Training and Validation Loss',
                                 color_discrete_sequence=['#3B82F6', '#EF4444'])
                st.plotly_chart(fig_loss, width='stretch')
            
            with col_hist2:
                if 'train_accs' in checkpoint and 'val_accs' in checkpoint:
                    acc_df = pd.DataFrame({
                        'Epoch': list(epochs) * 2,
                        'Accuracy': checkpoint['train_accs'] + checkpoint['val_accs'],
                        'Type': ['Train'] * len(epochs) + ['Validation'] * len(epochs)
                    })
                    
                    fig_acc = px.line(acc_df, x='Epoch', y='Accuracy', color='Type',
                                     title='Training and Validation Accuracy',
                                     color_discrete_sequence=['#10B981', '#F59E0B'])
                    st.plotly_chart(fig_acc, width='stretch')
        
        # Dataset statistics
        st.markdown("---")
        st.markdown("### Dataset Statistics")
        
        data_dir = "data/hymenoptera_data"
        if os.path.exists(data_dir):
            stats = []
            
            for split in ['train', 'val', 'test']:
                for class_name in ['ants', 'bees']:
                    class_path = os.path.join(data_dir, split, class_name)
                    if os.path.exists(class_path):
                        num_images = len([f for f in os.listdir(class_path) 
                                        if f.endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
                        stats.append({
                            'Split': split.capitalize(),
                            'Class': class_name,
                            'Count': num_images
                        })
            
            if stats:
                stats_df = pd.DataFrame(stats)
                
                col_stat1, col_stat2 = st.columns(2)
                
                with col_stat1:
                    # Bar chart
                    fig_bar = px.bar(stats_df, x='Split', y='Count', color='Class',
                                   barmode='group', title='Image Distribution by Split and Class',
                                   color_discrete_sequence=['#3B82F6', '#10B981'])
                    st.plotly_chart(fig_bar, width='stretch')
                
                with col_stat2:
                    # Pie chart for training data
                    train_stats = stats_df[stats_df['Split'] == 'Train']
                    fig_pie = px.pie(train_stats, values='Count', names='Class',
                                   title='Training Data Distribution',
                                   color_discrete_sequence=['#3B82F6', '#10B981'])
                    st.plotly_chart(fig_pie, width='stretch')
        
        # Model information
        st.markdown("---")
        st.markdown("### Model Information")
        
        col_info1, col_info2 = st.columns(2)
        
        with col_info1:
            st.markdown("""
            **Model Details:**
            - Architecture: ResNeXt101
            - Input Size: 224x224
            - Number of Classes: 2
            - Pretrained: Yes
            - Feature Extraction: Frozen base
            """)
        
        with col_info2:
            if 'train_accs' in checkpoint and 'val_accs' in checkpoint:
                final_train_acc = checkpoint['train_accs'][-1] if checkpoint['train_accs'] else 0
                final_val_acc = checkpoint['val_accs'][-1] if checkpoint['val_accs'] else 0
                
                st.markdown("""
                **Performance Metrics:**
                - Final Training Accuracy: {:.2%}
                - Final Validation Accuracy: {:.2%}
                - Total Training Epochs: {}
                """.format(final_train_acc, final_val_acc, len(checkpoint['train_losses'])))
    
    def about_page(self):
        """About page"""
        st.markdown('<h2 class="sub-header">📖 About This Application</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            ### Explainable ML Vision Application
            
            This application demonstrates state-of-the-art explainable AI techniques
            for computer vision models. It provides a comprehensive suite of tools
            for training, analyzing, and interpreting deep learning models.
            
            ### Technologies Used:
            
            **Backend & ML:**
            - PyTorch for deep learning
            - TorchVision for computer vision models
            - SHAP for SHapley Additive exPlanations
            - LIME for local interpretable explanations
            - Grad-CAM for visual explanations
            
            **Frontend & Visualization:**
            - Streamlit for web application
            - Plotly for interactive charts
            - Matplotlib & Seaborn for static visualizations
            
            ### Key Features:
            
            1. **Model Training Interface**
               - Multiple model architectures
               - Customizable hyperparameters
               - Real-time training monitoring
            
            2. **Explainability Suite**
               - Grad-CAM: Visual attention maps
               - LIME: Local feature importance
               - SHAP: Global feature attribution
            
            3. **Comprehensive Analysis**
               - Training history visualization
               - Confusion matrices
               - Class distribution analysis
               - Performance metrics
            
            ### Use Cases:
            
            - **Researchers**: Study model behavior and interpretability
            - **Developers**: Debug and improve model performance
            - **Students**: Learn about explainable AI techniques
            - **Business Users**: Understand AI decision-making
            
            ### Dataset:
            
            The application uses the Hymenoptera dataset (ants vs bees) from
            PyTorch tutorials. Users can replace this with their own dataset
            following the same directory structure.
            """)
        
        with col2:
            st.markdown("### Quick Links")
            
            st.markdown("""
            [![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
            [![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
            [![SHAP](https://img.shields.io/badge/SHAP-FF6B6B?style=for-the-badge)](https://shap.readthedocs.io)
            [![LIME](https://img.shields.io/badge/LIME-00C853?style=for-the-badge)](https://github.com/marcotcr/lime)
            """)
            
            st.markdown("### Contact & Support")
            
            st.markdown("""
            **For support or questions:**
            - Check the GitHub repository
            - Open an issue for bugs
            - Submit feature requests
            
            **Contributing:**
            Contributions are welcome! Please fork the repository
            and submit pull requests.
            """)

# Run the application
if __name__ == "__main__":
    app = ExplainableMLApp()
    app.run()