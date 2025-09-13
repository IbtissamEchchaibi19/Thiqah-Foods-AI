# components/model_performance.py
"""
Model performance tab component for the Thiqah Foods AI application
"""

import streamlit as st
import plotly.express as px
import pandas as pd
from utils.helpers import apply_dark_theme


def render_model_performance_tab(df, predictor, model_metrics):
    """Render the model performance and metrics tab"""
    st.header("Model Performance")
    
    # Model metrics overview
    render_model_metrics_overview(model_metrics)
    
    # Model details and configuration
    render_model_details(predictor)
    
    # Performance visualizations
    render_performance_visualizations(df, model_metrics)
    
    # Training data analysis
    render_training_data_analysis(df)


def render_model_metrics_overview(model_metrics):
    """Render overview of model performance metrics"""
    st.subheader("Model Performance Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Regression R² Score", 
            f"{model_metrics.get('regression_r2', 0):.3f}",
            help="Coefficient of determination for risk score prediction"
        )
    
    with col2:
        st.metric(
            "Classification Accuracy", 
            f"{model_metrics.get('classification_accuracy', 0):.3f}",
            help="Accuracy for risk category classification"
        )
    
    with col3:
        n_features = model_metrics.get('n_features', 0)
        st.metric(
            "Number of Features", 
            n_features,
            help="Total number of features used in the model"
        )
    
    with col4:
        n_train_samples = model_metrics.get('n_train_samples', 0)
        st.metric(
            "Training Samples", 
            f"{n_train_samples:,}",
            help="Number of samples used for training"
        )


def render_model_details(predictor):
    """Render detailed model information"""
    st.subheader("Model Architecture")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### Model Components
        
        **Risk Score Prediction:**
        - Algorithm: Random Forest Regressor
        - Estimators: 200 trees
        - Max Depth: 15
        - Features: 18 engineered features
        
        **Risk Category Classification:**
        - Algorithm: Gradient Boosting Classifier
        - Estimators: 200 boosting stages
        - Max Depth: 6
        - Learning Rate: 0.1
        """)
    
    with col2:
        if predictor.is_trained:
            model_summary = predictor.model_summary()
            st.markdown("### Model Status")
            st.write(f"**Training Status:** ✅ Trained")
            st.write(f"**Features Used:** {model_summary.get('n_features', 'N/A')}")
            st.write(f"**Encoders:** {len(model_summary.get('encoders', []))}")
            st.write(f"**Scalers:** {len(model_summary.get('scalers', []))}")
        else:
            st.markdown("### Model Status")
            st.write("**Training Status:** ❌ Not Trained")


def render_performance_visualizations(df, model_metrics):
    """Render performance visualization charts"""
    st.subheader("Performance Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        render_training_metrics_chart(model_metrics)
    
    with col2:
        render_feature_importance_summary(model_metrics)
    
    # Model accuracy breakdown
    render_accuracy_breakdown(df)


def render_training_metrics_chart(model_metrics):
    """Render training metrics as a chart"""
    try:
        metrics_data = {
            'Metric': ['R² Score', 'Classification Accuracy'],
            'Value': [
                model_metrics.get('regression_r2', 0),
                model_metrics.get('classification_accuracy', 0)
            ],
            'Target': [0.85, 0.90]  # Target performance values
        }
        
        metrics_df = pd.DataFrame(metrics_data)
        
        fig = px.bar(
            metrics_df, 
            x='Metric', 
            y='Value',
            title="Model Performance vs Targets",
            color='Value',
            color_continuous_scale='Viridis'
        )
        
        # Add target line
        fig.add_hline(y=0.85, line_dash="dash", line_color="red", 
                      annotation_text="Target Performance")
        
        fig = apply_dark_theme(fig)
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error rendering training metrics: {str(e)}")


def render_feature_importance_summary(model_metrics):
    """Render a summary of top feature importance"""
    try:
        feature_importance = model_metrics.get('feature_importance', {})
        
        if feature_importance:
            # Get top 5 features
            top_features = dict(sorted(
                feature_importance.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:5])
            
            fig = px.pie(
                values=list(top_features.values()),
                names=list(top_features.keys()),
                title="Top 5 Feature Importance"
            )
            
            fig = apply_dark_theme(fig)
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Feature importance data not available")
            
    except Exception as e:
        st.error(f"Error rendering feature importance: {str(e)}")


def render_accuracy_breakdown(df):
    """Render accuracy breakdown by different categories"""
    st.subheader("Model Accuracy Breakdown")
    
    try:
        # Accuracy by risk category
        risk_category_counts = df['risk_category'].value_counts()
        
        accuracy_data = {
            'Risk Category': risk_category_counts.index,
            'Sample Count': risk_category_counts.values,
            'Percentage': (risk_category_counts.values / len(df) * 100).round(1)
        }
        
        accuracy_df = pd.DataFrame(accuracy_data)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Risk Category Distribution in Training Data:**")
            st.dataframe(accuracy_df, use_container_width=True)
        
        with col2:
            fig = px.pie(
                accuracy_df,
                values='Sample Count',
                names='Risk Category',
                title="Training Data Distribution",
                color_discrete_sequence=['#4caf50', '#ff9800', '#f44336']
            )
            
            fig = apply_dark_theme(fig)
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)
            
    except Exception as e:
        st.error(f"Error rendering accuracy breakdown: {str(e)}")


def render_training_data_analysis(df):
    """Render analysis of the training data used"""
    st.subheader("Training Data Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        render_data_quality_metrics(df)
    
    with col2:
        render_data_distribution_analysis(df)
    
    # Feature statistics
    render_feature_statistics(df)


def render_data_quality_metrics(df):
    """Render data quality metrics"""
    st.markdown("### Data Quality Metrics")
    
    try:
        # Calculate data quality metrics
        total_samples = len(df)
        missing_values = df.isnull().sum().sum()
        complete_cases = len(df.dropna())
        
        quality_metrics = {
            'Total Samples': f"{total_samples:,}",
            'Complete Cases': f"{complete_cases:,}",
            'Missing Values': missing_values,
            'Data Completeness': f"{(complete_cases/total_samples)*100:.1f}%"
        }
        
        for metric, value in quality_metrics.items():
            st.write(f"**{metric}:** {value}")
            
    except Exception as e:
        st.error(f"Error calculating data quality metrics: {str(e)}")


def render_data_distribution_analysis(df):
    """Render data distribution analysis"""
    st.markdown("### Data Distribution")
    
    try:
        # Risk score distribution
        fig = px.histogram(
            df, 
            x='risk_score', 
            nbins=30, 
            title="Risk Score Distribution in Training Data",
            color_discrete_sequence=['#4caf50']
        )
        
        fig = apply_dark_theme(fig)
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error rendering data distribution: {str(e)}")


def render_feature_statistics(df):
    """Render feature statistics table"""
    st.subheader("Feature Statistics")
    
    try:
        # Select numeric features for statistics
        numeric_features = [
            'temperature', 'humidity', 'cold_chain_temp', 'transport_duration',
            'storage_temp', 'storage_duration', 'mycotoxin_level', 'risk_score'
        ]
        
        feature_stats = df[numeric_features].describe().round(2)
        
        # Display statistics table
        st.dataframe(feature_stats, use_container_width=True)
        
        # Feature ranges visualization
        render_feature_ranges_chart(df, numeric_features)
        
    except Exception as e:
        st.error(f"Error rendering feature statistics: {str(e)}")


def render_feature_ranges_chart(df, features):
    """Render feature ranges as box plots"""
    try:
        # Create box plots for feature ranges
        fig = px.box(
            df[features], 
            title="Feature Value Ranges",
            labels={'variable': 'Feature', 'value': 'Value'}
        )
        
        fig = apply_dark_theme(fig)
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error rendering feature ranges: {str(e)}")


def render_model_validation_info(model_metrics):
    """Render model validation and testing information"""
    st.subheader("Model Validation")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### Validation Approach
        - **Split Method:** Stratified train-test split
        - **Test Size:** 20% of total data
        - **Cross-Validation:** K-fold (k=5)
        - **Metrics:** R², Accuracy, Precision, Recall
        """)
    
    with col2:
        st.markdown("""
        ### Performance Benchmarks
        - **Target R² Score:** ≥ 0.85
        - **Target Accuracy:** ≥ 0.90
        - **Acceptable Performance:** ≥ 0.80
        - **Retraining Trigger:** < 0.75
        """)
    
    # Performance status
    regression_r2 = model_metrics.get('regression_r2', 0)
    classification_acc = model_metrics.get('classification_accuracy', 0)
    
    if regression_r2 >= 0.85 and classification_acc >= 0.85:
        st.success("✅ Model performance meets target benchmarks")
    elif regression_r2 >= 0.75 and classification_acc >= 0.75:
        st.warning("⚠️ Model performance is acceptable but below targets")
    else:
        st.error("❌ Model performance is below acceptable thresholds")


def render_improvement_recommendations():
    """Render recommendations for model improvement"""
    st.subheader("Model Improvement Recommendations")
    
    recommendations = [
        "**Feature Engineering**: Consider adding more domain-specific features like seasonal patterns",
        "**Data Augmentation**: Collect more samples from underrepresented regions or food types",
        "**Model Ensemble**: Combine multiple algorithms for improved prediction accuracy",
        "**Hyperparameter Tuning**: Optimize model parameters using grid search or Bayesian optimization",
        "**Real-time Learning**: Implement online learning to adapt to new patterns",
        "**External Data**: Incorporate weather data, supply chain delays, or regulatory changes"
    ]
    
    for i, rec in enumerate(recommendations, 1):
        st.markdown(f"{i}. {rec}")
    
    # Performance monitoring section
    st.markdown("""
    ### Continuous Monitoring
    
    The model performance should be continuously monitored using:
    - **Drift Detection**: Monitor for changes in input data distribution
    - **Performance Tracking**: Regular evaluation on new validation datasets  
    - **Feedback Integration**: Incorporate user feedback on prediction quality
    - **Automated Retraining**: Schedule periodic model retraining with new data
    """)