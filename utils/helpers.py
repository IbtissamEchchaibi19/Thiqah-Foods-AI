
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import random

from config.settings import (
    REGIONS, FOOD_TYPES, SUPPLY_CHAIN_STAGES, SEASONS,
    SEASONAL_TEMPS, IDEAL_CONDITIONS, COLORS
)


def apply_dark_theme(fig):
    """Apply dark theme styling to Plotly figures"""
    fig.update_layout(
        paper_bgcolor=COLORS['dark'],
        plot_bgcolor=COLORS['background'],
        font=dict(color='#fafafa'),
        xaxis=dict(
            gridcolor='#2d3436', 
            tickcolor='#fafafa', 
            linecolor='#2d3436'
        ),
        yaxis=dict(
            gridcolor='#2d3436', 
            tickcolor='#fafafa', 
            linecolor='#2d3436'
        )
    )
    return fig


def prepare_data(df):
    """Prepare and enhance dataset with feature engineering"""
    # Feature engineering
    df['temp_humidity_interaction'] = df['temperature'] * df['humidity'] / 100
    df['cold_chain_deviation'] = abs(df['cold_chain_temp'] - IDEAL_CONDITIONS['cold_chain_temp'])
    df['storage_risk_factor'] = df['storage_temp'] * df['storage_duration']
    df['total_duration'] = df['transport_duration'] + df['storage_duration'] * 24
    
    return df


def calculate_risk_factors(row):
    """Calculate risk factors for a given data row"""
    risk_factors = []
    
    # Cold chain analysis
    if row['cold_chain_temp'] > IDEAL_CONDITIONS['cold_chain_temp_range']['max'] or \
       row['cold_chain_temp'] < IDEAL_CONDITIONS['cold_chain_temp_range']['min']:
        risk_factors.append('Cold chain temperature deviation')
    
    # High humidity risk
    if row['humidity'] > IDEAL_CONDITIONS['high_humidity_threshold']:
        risk_factors.append('High humidity conditions')
    
    # Extended transport
    if row['transport_duration'] > IDEAL_CONDITIONS['max_transport_duration']:
        risk_factors.append('Extended transport duration')
    
    # Pathogen detection
    if row['pathogen_detected']:
        risk_factors.append('Pathogen contamination detected')
    
    # Mycotoxin levels
    if row['mycotoxin_level'] > IDEAL_CONDITIONS['mycotoxin_safe_level']:
        risk_factors.append('High mycotoxin levels')
    
    return risk_factors


def format_metric_card(title, value, delta=None, color='primary'):
    """Format metric card HTML"""
    color_class = f"risk-{color.lower()}" if color in ['high', 'medium', 'low'] else ''
    delta_html = f"<small>{delta}</small>" if delta else ""
    
    return f"""
    <div class="metric-container {color_class}">
        <h3>{title}</h3>
        <h2>{value}</h2>
        {delta_html}
    </div>
    """


def generate_alert_message(alert_data):
    """Generate formatted alert message"""
    risk_class = f"risk-{alert_data['risk_category'].lower()}"
    
    return f"""
    <div class="metric-container {risk_class}">
        <strong>{alert_data['region']} - {alert_data['food_type']}</strong><br>
        Risk Score: {alert_data['risk_score']:.1f} | 
        Stage: {alert_data['supply_chain_stage']} | 
        Date: {alert_data['date'].strftime('%Y-%m-%d')}
    </div>
    """


def get_risk_color_and_icon(risk_score):
    """Get appropriate color and icon for risk score"""
    if risk_score >= 70:
        return COLORS['danger'], "🔴"
    elif risk_score >= 40:
        return COLORS['warning'], "🟡"
    else:
        return COLORS['success'], "🟢"


def format_prediction_result(prediction):
    """Format prediction result for display"""
    color, icon = get_risk_color_and_icon(prediction['risk_score'])
    
    result = {
        'score': prediction['risk_score'],
        'category': prediction['risk_category'],
        'icon': icon,
        'color': color,
        'confidence': prediction.get('confidence', 0),
        'probabilities': prediction['risk_probabilities']
    }
    
    return result


def validate_input_data(input_dict):
    """Validate user input data"""
    errors = []
    
    # Temperature validation
    if not -10 <= input_dict.get('temperature', 0) <= 60:
        errors.append("Temperature must be between -10°C and 60°C")
    
    # Humidity validation
    if not 0 <= input_dict.get('humidity', 0) <= 100:
        errors.append("Humidity must be between 0% and 100%")
    
    # Cold chain temperature validation
    if not -10 <= input_dict.get('cold_chain_temp', 0) <= 15:
        errors.append("Cold chain temperature must be between -10°C and 15°C")
    
    # Duration validations
    if not 0 <= input_dict.get('transport_duration', 0) <= 168:  # 7 days max
        errors.append("Transport duration must be between 0 and 168 hours")
    
    if not 0 <= input_dict.get('storage_duration', 0) <= 365:  # 1 year max
        errors.append("Storage duration must be between 0 and 365 days")
    
    # Mycotoxin validation
    if not 0 <= input_dict.get('mycotoxin_level', 0) <= 1000:
        errors.append("Mycotoxin level must be between 0 and 1000 ppb")
    
    return errors


def get_seasonal_recommendations(season, food_type):
    """Get seasonal recommendations for food safety"""
    recommendations = []
    
    if season == 'Summer':
        recommendations.extend([
            "Increase cold chain monitoring due to high temperatures",
            "Reduce storage duration for perishable items",
            "Monitor humidity levels more frequently"
        ])
        
        if food_type in ['Dairy', 'Meat', 'Seafood']:
            recommendations.append("Extra vigilance required for protein-rich foods")
    
    elif season == 'Winter':
        recommendations.extend([
            "Monitor for condensation in storage areas",
            "Ensure adequate ventilation in storage facilities"
        ])
    
    # Food-specific recommendations
    if food_type == 'Vegetables':
        recommendations.append("Monitor for moisture content and spoilage")
    elif food_type == 'Grains':
        recommendations.append("Check for mycotoxin contamination regularly")
    elif food_type == 'Seafood':
        recommendations.append("Maintain strict cold chain throughout transport")
    
    return recommendations


def calculate_trend_direction(values):
    """Calculate trend direction for time series data"""
    if len(values) < 2:
        return "stable"
    
    recent_avg = np.mean(values[-3:])
    previous_avg = np.mean(values[-6:-3]) if len(values) >= 6 else np.mean(values[:-3])
    
    if recent_avg > previous_avg * 1.1:
        return "increasing"
    elif recent_avg < previous_avg * 0.9:
        return "decreasing"
    else:
        return "stable"


def generate_summary_stats(df):
    """Generate summary statistics for the dataset"""
    stats = {
        'total_samples': len(df),
        'high_risk_count': len(df[df['risk_category'] == 'High']),
        'medium_risk_count': len(df[df['risk_category'] == 'Medium']),
        'low_risk_count': len(df[df['risk_category'] == 'Low']),
        'avg_risk_score': df['risk_score'].mean(),
        'pathogen_detection_rate': df['pathogen_detected'].mean() * 100,
        'cold_chain_break_rate': df['cold_chain_break'].mean() * 100,
        'avg_transport_duration': df['transport_duration'].mean(),
        'avg_storage_duration': df['storage_duration'].mean()
    }
    
    # Add percentages
    stats['high_risk_percentage'] = (stats['high_risk_count'] / stats['total_samples']) * 100
    stats['medium_risk_percentage'] = (stats['medium_risk_count'] / stats['total_samples']) * 100
    stats['low_risk_percentage'] = (stats['low_risk_count'] / stats['total_samples']) * 100
    
    return stats


def export_prediction_report(prediction_data, input_data):
    """Export prediction results as formatted report"""
    report = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'input_parameters': input_data.to_dict('records')[0],
        'prediction_results': prediction_data,
        'recommendations': prediction_data.get('risk_assessment', {}).get('recommendations', []),
        'critical_factors': prediction_data.get('risk_assessment', {}).get('critical_factors', [])
    }
    
    return report