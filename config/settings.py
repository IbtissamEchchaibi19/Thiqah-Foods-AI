# Constants, regions, food types, model params
# config/settings.py
"""
Configuration settings for Thiqah Foods AI Risk Prediction System
"""

# Application Configuration
APP_CONFIG = {
    'title': "Thiqah Foods AI Platform (ثقة)",
    'subtitle': "Advanced Food Safety Risk Prediction System",
    'page_icon': "🛡️",
    'layout': "wide",
    'sidebar_state': "expanded"
}

# Data Configuration
DATA_CONFIG = {
    'n_samples': 5000,
    'random_seed': 42,
    'history_years': 3,
    'test_size': 0.2
}

# Regional Data
REGIONS = [
    'Riyadh', 'Jeddah', 'Dammam', 'Mecca', 
    'Medina', 'Tabuk', 'Abha', 'Hail'
]

FOOD_TYPES = [
    'Dairy', 'Meat', 'Vegetables', 'Fruits', 
    'Grains', 'Seafood', 'Processed'
]

SUPPLY_CHAIN_STAGES = [
    'Farm', 'Processing', 'Storage', 'Transport', 'Retail'
]

SEASONS = ['Spring', 'Summer', 'Fall', 'Winter']

# Risk Thresholds
RISK_THRESHOLDS = {
    'low': {'min': 0, 'max': 39},
    'medium': {'min': 40, 'max': 69},
    'high': {'min': 70, 'max': 100}
}

# Model Configuration
MODEL_CONFIG = {
    'random_forest': {
        'n_estimators': 200,
        'max_depth': 15,
        'random_state': 42,
        'n_jobs': -1
    },
    'gradient_boosting': {
        'n_estimators': 200,
        'max_depth': 6,
        'learning_rate': 0.1,
        'random_state': 42
    }
}

# Feature Configuration
FEATURE_COLUMNS = [
    'temperature', 'humidity', 'cold_chain_temp', 'cold_chain_humidity',
    'transport_duration', 'storage_temp', 'storage_humidity', 'storage_duration',
    'pathogen_detected', 'mycotoxin_level', 'temp_humidity_interaction',
    'cold_chain_deviation', 'storage_risk_factor', 'total_duration'
]

CATEGORICAL_COLUMNS = ['region', 'food_type', 'supply_chain_stage', 'season']

# Temperature Ranges by Season (°C)
SEASONAL_TEMPS = {
    'Summer': {'min': 35, 'max': 48},
    'Winter': {'min': 15, 'max': 25},
    'Spring': {'min': 20, 'max': 35},
    'Fall': {'min': 20, 'max': 35}
}

# Ideal Storage Conditions
IDEAL_CONDITIONS = {
    'cold_chain_temp': 2,  # °C
    'cold_chain_temp_range': {'min': -1, 'max': 4},
    'max_transport_duration': 24,  # hours
    'high_humidity_threshold': 70,  # %
    'mycotoxin_safe_level': 30  # ppb
}

# UI Colors and Styling
COLORS = {
    'primary': '#2d5a3d',
    'success': '#4caf50',
    'warning': '#ff9800',
    'danger': '#f44336',
    'info': '#2196f3',
    'dark': '#1e2329',
    'background': '#0e1117'
}

# Chart Color Palettes
COLOR_PALETTES = {
    'risk_categories': ['#4caf50', '#ff9800', '#f44336'],
    'default': ['#4caf50', '#2196f3', '#ff9800', '#f44336', '#9c27b0', '#00bcd4', '#8bc34a', '#ffc107']
}

# Dashboard Metrics Configuration
DASHBOARD_METRICS = {
    'update_interval': 30,  # seconds
    'max_alerts': 5,
    'chart_height': 400
}