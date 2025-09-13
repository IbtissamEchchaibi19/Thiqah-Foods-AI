
import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np
from utils.helpers import apply_dark_theme
from config.settings import COLOR_PALETTES


def render_analytics_tab(df, predictor):
    """Render the advanced analytics tab"""
    st.header("Advanced Analytics")
    
    # Feature importance analysis
    render_feature_importance(predictor)
    
    # Correlation analysis
    render_correlation_analysis(df)
    
    # Risk analysis by different factors
    col1, col2 = st.columns(2)
    
    with col1:
        render_risk_by_food_type(df)
    
    with col2:
        render_risk_by_season(df)
    
    # Advanced statistical analysis
    render_statistical_analysis(df)


def render_feature_importance(predictor):
    """Render feature importance analysis"""
    st.subheader("Feature Importance Analysis")
    
    if predictor.is_trained:
        try:
            # Get feature importance
            importance_dict = predictor.get_feature_importance(top_n=10)
            
            if importance_dict:
                # Convert to DataFrame for plotting
                importance_df = pd.DataFrame([
                    {'feature': k, 'importance': v} 
                    for k, v in importance_dict.items()
                ]).sort_values('importance', ascending=True)
                
                # Create horizontal bar chart
                fig = px.bar(
                    importance_df, 
                    x='importance', 
                    y='feature', 
                    orientation='h',
                    title="Top 10 Most Important Features",
                    color='importance', 
                    color_continuous_scale='viridis',
                    labels={'importance': 'Feature Importance', 'feature': 'Feature'}
                )
                
                fig = apply_dark_theme(fig)
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
                
                # Feature importance table
                st.subheader("Feature Importance Scores")
                st.dataframe(importance_df.sort_values('importance', ascending=False), use_container_width=True)
                
            else:
                st.warning("Feature importance data not available")
                
        except Exception as e:
            st.error(f"Error rendering feature importance: {str(e)}")
    else:
        st.warning("Model not trained yet. Feature importance will be available after training.")


def render_correlation_analysis(df):
    """Render feature correlation heatmap"""
    st.subheader("Feature Correlation Analysis")
    
    try:
        # Select numeric columns for correlation analysis
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        corr_matrix = df[numeric_cols].corr()
        
        # Create correlation heatmap
        fig = px.imshow(
            corr_matrix, 
            text_auto=True, 
            aspect="auto",
            title="Feature Correlation Heatmap",
            color_continuous_scale='RdBu_r',
            zmin=-1, zmax=1
        )
        
        fig = apply_dark_theme(fig)
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)
        
        # Correlation insights
        render_correlation_insights(corr_matrix)
        
    except Exception as e:
        st.error(f"Error rendering correlation analysis: {str(e)}")


def render_correlation_insights(corr_matrix):
    """Render insights from correlation analysis"""
    st.subheader("Correlation Insights")
    
    try:
        # Find highest correlations with risk_score
        if 'risk_score' in corr_matrix.columns:
            risk_correlations = corr_matrix['risk_score'].abs().sort_values(ascending=False)[1:6]
            
            st.markdown("**Top 5 Features Most Correlated with Risk Score:**")
            for feature, correlation in risk_correlations.items():
                correlation_strength = "Strong" if abs(correlation) > 0.7 else "Moderate" if abs(correlation) > 0.4 else "Weak"
                st.write(f"• {feature}: {correlation:.3f} ({correlation_strength})")
        
        # Find highly correlated feature pairs
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.7:
                    high_corr_pairs.append((
                        corr_matrix.columns[i], 
                        corr_matrix.columns[j], 
                        corr_val
                    ))
        
        if high_corr_pairs:
            st.markdown("**Highly Correlated Feature Pairs (|r| > 0.7):**")
            for feat1, feat2, corr_val in high_corr_pairs[:5]:
                st.write(f"• {feat1} ↔ {feat2}: {corr_val:.3f}")
                
    except Exception as e:
        st.error(f"Error generating correlation insights: {str(e)}")


def render_risk_by_food_type(df):
    """Render risk analysis by food type"""
    try:
        # Calculate average risk by food type
        risk_by_food = df.groupby('food_type')['risk_score'].mean().sort_values(ascending=False)
        
        fig = px.bar(
            x=risk_by_food.index,
            y=risk_by_food.values,
            title="Average Risk Score by Food Type",
            color=risk_by_food.values,
            color_continuous_scale='Reds',
            labels={'x': 'Food Type', 'y': 'Average Risk Score'}
        )
        
        fig = apply_dark_theme(fig)
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error rendering food type analysis: {str(e)}")


def render_risk_by_season(df):
    """Render risk analysis by season"""
    try:
        # Calculate average risk by season
        risk_by_season = df.groupby('season')['risk_score'].mean()
        
        fig = px.bar(
            x=risk_by_season.index,
            y=risk_by_season.values,
            title="Average Risk Score by Season",
            color=risk_by_season.values,
            color_continuous_scale='Blues',
            labels={'x': 'Season', 'y': 'Average Risk Score'}
        )
        
        fig = apply_dark_theme(fig)
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error rendering seasonal analysis: {str(e)}")


def render_statistical_analysis(df):
    """Render detailed statistical analysis"""
    st.subheader("Statistical Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        render_risk_distribution(df)
    
    with col2:
        render_pathogen_analysis(df)
    
    # Supply chain stage analysis
    render_supply_chain_analysis(df)
    
    # Temperature vs humidity analysis
    render_temperature_humidity_analysis(df)


def render_risk_distribution(df):
    """Render risk score distribution analysis"""
    try:
        # Risk score distribution
        fig = px.histogram(
            df, 
            x='risk_score', 
            nbins=30, 
            title="Risk Score Distribution",
            color_discrete_sequence=['#4caf50'],
            labels={'x': 'Risk Score', 'y': 'Frequency'}
        )
        
        fig = apply_dark_theme(fig)
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)
        
        # Statistical summary
        st.markdown("**Risk Score Statistics:**")
        st.write(f"Mean: {df['risk_score'].mean():.2f}")
        st.write(f"Median: {df['risk_score'].median():.2f}")
        st.write(f"Std Dev: {df['risk_score'].std():.2f}")
        
    except Exception as e:
        st.error(f"Error rendering risk distribution: {str(e)}")


def render_pathogen_analysis(df):
    """Render pathogen detection analysis"""
    try:
        # Pathogen detection by food type
        pathogen_by_food = df.groupby('food_type')['pathogen_detected'].mean() * 100
        
        fig = px.bar(
            x=pathogen_by_food.index,
            y=pathogen_by_food.values,
            title="Pathogen Detection Rate by Food Type (%)",
            color=pathogen_by_food.values,
            color_continuous_scale='Oranges',
            labels={'x': 'Food Type', 'y': 'Detection Rate (%)'}
        )
        
        fig = apply_dark_theme(fig)
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error rendering pathogen analysis: {str(e)}")


def render_supply_chain_analysis(df):
    """Render supply chain stage analysis"""
    try:
        st.subheader("Supply Chain Stage Analysis")
        
        # Risk by supply chain stage
        stage_analysis = df.groupby('supply_chain_stage').agg({
            'risk_score': 'mean',
            'pathogen_detected': lambda x: (x == 1).sum(),
            'cold_chain_break': 'sum',
            'transport_duration': 'mean'
        }).round(2)
        
        stage_analysis.columns = ['Avg Risk Score', 'Pathogen Detections', 'Cold Chain Breaks', 'Avg Transport Duration']
        
        st.dataframe(stage_analysis, use_container_width=True)
        
        # Visualization
        fig = px.scatter(
            df, 
            x='transport_duration', 
            y='risk_score', 
            color='supply_chain_stage',
            size='mycotoxin_level',
            title="Risk Score vs Transport Duration by Supply Chain Stage",
            labels={'transport_duration': 'Transport Duration (hours)', 'risk_score': 'Risk Score'}
        )
        
        fig = apply_dark_theme(fig)
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error rendering supply chain analysis: {str(e)}")


def render_temperature_humidity_analysis(df):
    """Render temperature vs humidity analysis"""
    try:
        st.subheader("Temperature vs Humidity Analysis")
        
        # Scatter plot of temperature vs humidity colored by risk
        fig = px.scatter(
            df, 
            x='temperature', 
            y='humidity', 
            color='risk_score',
            size='mycotoxin_level',
            facet_col='season',
            title="Temperature vs Humidity by Season (Colored by Risk Score)",
            color_continuous_scale='Reds',
            labels={'temperature': 'Temperature (°C)', 'humidity': 'Humidity (%)'}
        )
        
        fig = apply_dark_theme(fig)
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        # Risk zones analysis
        render_risk_zones_analysis(df)
        
    except Exception as e:
        st.error(f"Error rendering temperature humidity analysis: {str(e)}")


def render_risk_zones_analysis(df):
    """Render risk zones analysis based on temperature and humidity"""
    try:
        # Define risk zones
        df['temp_zone'] = pd.cut(df['temperature'], bins=[0, 15, 25, 35, 50], labels=['Cold', 'Cool', 'Warm', 'Hot'])
        df['humidity_zone'] = pd.cut(df['humidity'], bins=[0, 30, 50, 70, 100], labels=['Low', 'Medium', 'High', 'Very High'])
        
        # Calculate average risk by zones
        risk_zones = df.groupby(['temp_zone', 'humidity_zone'])['risk_score'].mean().unstack(fill_value=0)
        
        # Create heatmap
        fig = px.imshow(
            risk_zones,
            title="Risk Score Heatmap: Temperature vs Humidity Zones",
            color_continuous_scale='Reds',
            labels={'x': 'Humidity Zone', 'y': 'Temperature Zone', 'color': 'Average Risk Score'}
        )
        
        fig = apply_dark_theme(fig)
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error rendering risk zones analysis: {str(e)}")