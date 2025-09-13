# components/risk_map.py
"""
Risk mapping tab component for the Thiqah Foods AI application
"""

import streamlit as st
import plotly.express as px
import pandas as pd
from utils.helpers import apply_dark_theme
from config.settings import REGIONS, COLOR_PALETTES


def render_risk_map_tab(df):
    """Render the risk heatmap and geographic analysis tab"""
    st.header("Risk Heatmap")
    st.markdown("Geographic distribution of food safety risks across regions")
    
    # Regional risk summary
    render_regional_summary(df)
    
    # Risk distribution visualization
    render_risk_distribution_map(df)
    
    # Detailed regional analysis
    render_detailed_regional_analysis(df)


def render_regional_summary(df):
    """Render regional risk summary table"""
    st.subheader("Regional Risk Summary")
    
    try:
        # Calculate regional statistics
        region_risk = df.groupby('region').agg({
            'risk_score': 'mean',
            'risk_category': lambda x: (x == 'High').sum(),
            'pathogen_detected': 'sum',
            'cold_chain_break': 'sum',
            'mycotoxin_level': 'mean',
            'transport_duration': 'mean'
        }).round(2)
        
        region_risk.columns = [
            'Avg Risk Score', 
            'High Risk Count', 
            'Pathogen Detections', 
            'Cold Chain Breaks',
            'Avg Mycotoxin (ppb)',
            'Avg Transport Duration (hrs)'
        ]
        
        # Sort by average risk score
        region_risk = region_risk.sort_values('Avg Risk Score', ascending=False)
        
        # Color-code the dataframe based on risk levels
        st.dataframe(
            region_risk.style.background_gradient(
                subset=['Avg Risk Score'], 
                cmap='Reds'
            ),
            use_container_width=True
        )
        
        # Regional insights
        render_regional_insights(region_risk)
        
    except Exception as e:
        st.error(f"Error generating regional summary: {str(e)}")


def render_regional_insights(region_risk):
    """Render insights from regional analysis"""
    col1, col2, col3 = st.columns(3)
    
    with col1:
        highest_risk_region = region_risk['Avg Risk Score'].idxmax()
        highest_risk_score = region_risk['Avg Risk Score'].max()
        st.metric(
            "Highest Risk Region", 
            highest_risk_region,
            delta=f"{highest_risk_score:.1f} avg score"
        )
    
    with col2:
        lowest_risk_region = region_risk['Avg Risk Score'].idxmin()
        lowest_risk_score = region_risk['Avg Risk Score'].min()
        st.metric(
            "Lowest Risk Region", 
            lowest_risk_region,
            delta=f"{lowest_risk_score:.1f} avg score"
        )
    
    with col3:
        total_high_risk = region_risk['High Risk Count'].sum()
        st.metric(
            "Total High Risk Cases", 
            int(total_high_risk)
        )


def render_risk_distribution_map(df):
    """Render risk distribution visualization by region"""
    st.subheader("Risk Distribution by Region")
    
    try:
        # Create risk distribution chart
        fig = px.scatter(
            df, 
            x='temperature', 
            y='humidity', 
            size='risk_score', 
            color='risk_category',
            facet_col='region', 
            facet_col_wrap=4,
            title="Risk Distribution by Region (Temperature vs Humidity)",
            color_discrete_map={
                'Low': COLOR_PALETTES['risk_categories'][0], 
                'Medium': COLOR_PALETTES['risk_categories'][1], 
                'High': COLOR_PALETTES['risk_categories'][2]
            },
            labels={
                'temperature': 'Temperature (°C)',
                'humidity': 'Humidity (%)',
                'risk_category': 'Risk Category'
            }
        )
        
        fig = apply_dark_theme(fig)
        fig.update_layout(height=800)
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error rendering risk distribution map: {str(e)}")


def render_detailed_regional_analysis(df):
    """Render detailed analysis for each region"""
    st.subheader("Detailed Regional Analysis")
    
    # Region selector
    selected_region = st.selectbox("Select Region for Detailed Analysis", REGIONS)
    
    if selected_region:
        render_single_region_analysis(df, selected_region)


def render_single_region_analysis(df, region):
    """Render detailed analysis for a specific region"""
    try:
        # Filter data for selected region
        region_data = df[df['region'] == region]
        
        if len(region_data) == 0:
            st.warning(f"No data available for {region}")
            return
        
        col1, col2 = st.columns(2)
        
        with col1:
            render_region_metrics(region_data, region)
        
        with col2:
            render_region_charts(region_data, region)
        
        # Time series analysis for the region
        render_region_time_series(region_data, region)
        
        # Food type analysis for the region
        render_region_food_analysis(region_data, region)
        
    except Exception as e:
        st.error(f"Error analyzing region {region}: {str(e)}")


def render_region_metrics(region_data, region):
    """Render key metrics for a specific region"""
    st.markdown(f"### {region} Key Metrics")
    
    # Calculate metrics
    avg_risk = region_data['risk_score'].mean()
    high_risk_count = len(region_data[region_data['risk_category'] == 'High'])
    pathogen_rate = region_data['pathogen_detected'].mean() * 100
    cold_chain_breaks = region_data['cold_chain_break'].sum()
    
    st.metric("Average Risk Score", f"{avg_risk:.1f}")
    st.metric("High Risk Cases", high_risk_count)
    st.metric("Pathogen Detection Rate", f"{pathogen_rate:.1f}%")
    st.metric("Cold Chain Breaks", cold_chain_breaks)
    
    # Risk category distribution
    risk_dist = region_data['risk_category'].value_counts()
    st.markdown("**Risk Distribution:**")
    for category, count in risk_dist.items():
        percentage = (count / len(region_data)) * 100
        st.write(f"• {category}: {count} ({percentage:.1f}%)")


def render_region_charts(region_data, region):
    """Render charts for a specific region"""
    try:
        # Risk score distribution
        fig = px.histogram(
            region_data, 
            x='risk_score', 
            nbins=20, 
            title=f"Risk Score Distribution - {region}",
            color_discrete_sequence=['#4caf50']
        )
        
        fig = apply_dark_theme(fig)
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error rendering region charts: {str(e)}")


def render_region_time_series(region_data, region):
    """Render time series analysis for a specific region"""
    try:
        st.markdown(f"### {region} Risk Trends Over Time")
        
        # Convert date to datetime and group by month
        region_data['date'] = pd.to_datetime(region_data['date'])
        monthly_risk = region_data.groupby(region_data['date'].dt.to_period('M')).agg({
            'risk_score': 'mean',
            'pathogen_detected': 'sum',
            'cold_chain_break': 'sum'
        })
        
        # Create time series chart
        fig = px.line(
            x=monthly_risk.index.astype(str), 
            y=monthly_risk['risk_score'], 
            title=f"Monthly Risk Score Trends - {region}",
            labels={'x': 'Month', 'y': 'Average Risk Score'}
        )
        
        fig = apply_dark_theme(fig)
        fig.update_traces(line_color='#ff9800', line_width=3)
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error rendering time series for {region}: {str(e)}")


def render_region_food_analysis(region_data, region):
    """Render food type analysis for a specific region"""
    try:
        st.markdown(f"### {region} Food Type Risk Analysis")
        
        # Risk by food type in this region
        food_risk = region_data.groupby('food_type').agg({
            'risk_score': 'mean',
            'pathogen_detected': 'sum',
            'mycotoxin_level': 'mean'
        }).round(2)
        
        food_risk.columns = ['Avg Risk Score', 'Pathogen Count', 'Avg Mycotoxin Level']
        
        st.dataframe(food_risk, use_container_width=True)
        
        # Food type risk visualization
        fig = px.bar(
            x=food_risk.index,
            y=food_risk['Avg Risk Score'],
            title=f"Average Risk Score by Food Type - {region}",
            color=food_risk['Avg Risk Score'],
            color_continuous_scale='Reds'
        )
        
        fig = apply_dark_theme(fig)
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error rendering food analysis for {region}: {str(e)}")


def render_comparative_analysis(df):
    """Render comparative analysis across regions"""
    st.subheader("Cross-Regional Comparative Analysis")
    
    try:
        # Create comparison metrics
        comparison_data = []
        
        for region in REGIONS:
            region_data = df[df['region'] == region]
            if len(region_data) > 0:
                comparison_data.append({
                    'Region': region,
                    'Sample Count': len(region_data),
                    'Avg Risk Score': region_data['risk_score'].mean(),
                    'High Risk %': (region_data['risk_category'] == 'High').mean() * 100,
                    'Pathogen Rate %': region_data['pathogen_detected'].mean() * 100,
                    'Avg Temperature': region_data['temperature'].mean(),
                    'Avg Humidity': region_data['humidity'].mean()
                })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Display comparison table
        st.dataframe(
            comparison_df.round(2).style.background_gradient(
                subset=['Avg Risk Score', 'High Risk %'], 
                cmap='Reds'
            ),
            use_container_width=True
        )
        
        # Regional ranking
        render_regional_ranking(comparison_df)
        
    except Exception as e:
        st.error(f"Error rendering comparative analysis: {str(e)}")


def render_regional_ranking(comparison_df):
    """Render regional risk ranking"""
    st.markdown("### Regional Risk Ranking")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Highest risk regions
        highest_risk = comparison_df.nlargest(3, 'Avg Risk Score')[['Region', 'Avg Risk Score']]
        st.markdown("**Highest Risk Regions:**")
        for idx, row in highest_risk.iterrows():
            st.write(f"{idx + 1}. {row['Region']}: {row['Avg Risk Score']:.1f}")
    
    with col2:
        # Lowest risk regions
        lowest_risk = comparison_df.nsmallest(3, 'Avg Risk Score')[['Region', 'Avg Risk Score']]
        st.markdown("**Lowest Risk Regions:**")
        for idx, row in lowest_risk.iterrows():
            st.write(f"{idx + 1}. {row['Region']}: {row['Avg Risk Score']:.1f}")


def render_regional_recommendations(df, selected_region=None):
    """Render region-specific recommendations"""
    st.subheader("Regional Recommendations")
    
    if selected_region:
        region_data = df[df['region'] == selected_region]
        avg_risk = region_data['risk_score'].mean()
        
        st.markdown(f"### Recommendations for {selected_region}")
        
        if avg_risk >= 60:
            st.error("**High Priority Actions Required:**")
            recommendations = [
                "Implement enhanced monitoring protocols",
                "Increase inspection frequency",
                "Review and upgrade cold chain infrastructure",
                "Provide additional staff training on food safety",
                "Consider temporary restrictions on high-risk products"
            ]
        elif avg_risk >= 40:
            st.warning("**Moderate Priority Actions:**")
            recommendations = [
                "Strengthen existing monitoring procedures",
                "Review storage and transport conditions",
                "Increase pathogen testing frequency",
                "Implement preventive maintenance schedules"
            ]
        else:
            st.success("**Maintenance Actions:**")
            recommendations = [
                "Continue current monitoring protocols",
                "Maintain staff training programs",
                "Regular equipment calibration",
                "Seasonal review of procedures"
            ]
        
        for i, rec in enumerate(recommendations, 1):
            st.write(f"{i}. {rec}")
    else:
        st.info("Select a region above to view specific recommendations.")