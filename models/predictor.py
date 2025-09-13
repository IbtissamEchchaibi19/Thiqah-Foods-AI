

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

from config.settings import (
    MODEL_CONFIG, FEATURE_COLUMNS, CATEGORICAL_COLUMNS, 
    DATA_CONFIG, RISK_THRESHOLDS
)


class FoodSafetyPredictor:
    """Advanced ML predictor for food safety risk assessment"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        self.feature_columns = []
        self.is_trained = False
        
    def prepare_features(self, df):
        """Prepare features for ML models"""
        # Select relevant numerical features
        X = df[FEATURE_COLUMNS].copy()
        
        # Encode categorical variables
        for col in CATEGORICAL_COLUMNS:
            if col in df.columns:
                if col not in self.encoders:
                    self.encoders[col] = LabelEncoder()
                    X[f'{col}_encoded'] = self.encoders[col].fit_transform(df[col])
                else:
                    # Handle unseen categories
                    try:
                        X[f'{col}_encoded'] = self.encoders[col].transform(df[col])
                    except ValueError:
                        # Assign default value for unseen categories
                        X[f'{col}_encoded'] = 0
        
        # Store feature columns
        self.feature_columns = X.columns.tolist()
        return X
    
    def train_models(self, df):
        """Train multiple ML models for risk prediction"""
        try:
            # Prepare features and targets
            X = self.prepare_features(df)
            y_regression = df['risk_score']
            y_classification = df['risk_category']
            
            # Split data
            X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test = train_test_split(
                X, y_regression, y_classification, 
                test_size=DATA_CONFIG['test_size'], 
                random_state=DATA_CONFIG['random_seed'], 
                stratify=y_classification
            )
            
            # Scale features
            self.scalers['standard'] = StandardScaler()
            X_train_scaled = self.scalers['standard'].fit_transform(X_train)
            X_test_scaled = self.scalers['standard'].transform(X_test)
            
            # Train regression model (Risk Score Prediction)
            self.models['risk_score'] = RandomForestRegressor(**MODEL_CONFIG['random_forest'])
            self.models['risk_score'].fit(X_train, y_reg_train)
            
            # Train classification model (Risk Category Prediction)
            self.models['risk_category'] = GradientBoostingClassifier(**MODEL_CONFIG['gradient_boosting'])
            self.models['risk_category'].fit(X_train, y_clf_train)
            
            # Calculate performance metrics
            reg_score = self.models['risk_score'].score(X_test, y_reg_test)
            clf_predictions = self.models['risk_category'].predict(X_test)
            clf_accuracy = accuracy_score(y_clf_test, clf_predictions)
            
            # Get feature importance
            feature_importance = dict(zip(
                self.feature_columns, 
                self.models['risk_score'].feature_importances_
            ))
            
            self.is_trained = True
            
            return {
                'regression_r2': reg_score,
                'classification_accuracy': clf_accuracy,
                'feature_importance': feature_importance,
                'n_train_samples': len(X_train),
                'n_test_samples': len(X_test),
                'n_features': len(self.feature_columns)
            }
            
        except Exception as e:
            raise RuntimeError(f"Model training failed: {str(e)}")
    
    def predict(self, input_data):
        """Make predictions on new data"""
        if not self.is_trained:
            raise ValueError("Models must be trained before making predictions!")
        
        try:
            # Prepare features
            X = self.prepare_features(input_data)
            
            # Ensure we have the same features as training
            missing_features = set(self.feature_columns) - set(X.columns)
            if missing_features:
                for feature in missing_features:
                    X[feature] = 0  # Default value for missing features
            
            # Reorder columns to match training
            X = X[self.feature_columns]
            
            # Make predictions
            risk_score = self.models['risk_score'].predict(X)[0]
            risk_score = max(0, min(100, risk_score))  # Clip to valid range
            
            risk_category = self.models['risk_category'].predict(X)[0]
            risk_probabilities = self.models['risk_category'].predict_proba(X)[0]
            
            # Create probability dictionary
            prob_dict = dict(zip(
                self.models['risk_category'].classes_, 
                risk_probabilities
            ))
            
            # Generate risk assessment
            risk_assessment = self._generate_risk_assessment(risk_score, input_data.iloc[0])
            
            return {
                'risk_score': round(risk_score, 2),
                'risk_category': risk_category,
                'risk_probabilities': prob_dict,
                'risk_assessment': risk_assessment,
                'confidence': round(max(risk_probabilities) * 100, 1)
            }
            
        except Exception as e:
            raise RuntimeError(f"Prediction failed: {str(e)}")
    
    def _generate_risk_assessment(self, risk_score, input_row):
        """Generate detailed risk assessment with recommendations"""
        assessment = {
            'level': 'Low',
            'color': '🟢',
            'recommendations': [],
            'critical_factors': []
        }
        
        # Determine risk level
        if risk_score >= RISK_THRESHOLDS['high']['min']:
            assessment['level'] = 'High'
            assessment['color'] = '🔴'
        elif risk_score >= RISK_THRESHOLDS['medium']['min']:
            assessment['level'] = 'Medium'
            assessment['color'] = '🟡'
        
        # Analyze critical factors and generate recommendations
        if hasattr(input_row, 'cold_chain_temp'):
            if input_row['cold_chain_temp'] > 4 or input_row['cold_chain_temp'] < -1:
                assessment['critical_factors'].append('Cold chain temperature deviation')
                assessment['recommendations'].append('Maintain cold chain temperature between -1°C and 4°C')
        
        if hasattr(input_row, 'pathogen_detected') and input_row['pathogen_detected']:
            assessment['critical_factors'].append('Pathogen contamination detected')
            assessment['recommendations'].append('Immediate quarantine and testing required')
        
        if hasattr(input_row, 'transport_duration') and input_row['transport_duration'] > 24:
            assessment['critical_factors'].append('Extended transport duration')
            assessment['recommendations'].append('Reduce transport time to under 24 hours')
        
        if hasattr(input_row, 'mycotoxin_level') and input_row['mycotoxin_level'] > 30:
            assessment['critical_factors'].append('High mycotoxin levels')
            assessment['recommendations'].append('Test for mycotoxin contamination and consider rejection')
        
        # Add general recommendations based on risk level
        if assessment['level'] == 'High':
            assessment['recommendations'].extend([
                'Immediate intervention required',
                'Isolate affected products',
                'Conduct thorough inspection'
            ])
        elif assessment['level'] == 'Medium':
            assessment['recommendations'].extend([
                'Increase monitoring frequency',
                'Review storage conditions',
                'Implement preventive measures'
            ])
        else:
            assessment['recommendations'].append('Continue current monitoring protocols')
        
        return assessment
    
    def get_feature_importance(self, top_n=10):
        """Get top N most important features"""
        if not self.is_trained:
            return {}
        
        importance = dict(zip(
            self.feature_columns,
            self.models['risk_score'].feature_importances_
        ))
        
        # Sort by importance and return top N
        sorted_importance = dict(sorted(
            importance.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:top_n])
        
        return sorted_importance
    
    def model_summary(self):
        """Get summary of trained models"""
        if not self.is_trained:
            return "Models not trained yet"
        
        return {
            'regression_model': 'Random Forest Regressor',
            'classification_model': 'Gradient Boosting Classifier',
            'n_features': len(self.feature_columns),
            'feature_columns': self.feature_columns,
            'encoders': list(self.encoders.keys()),
            'scalers': list(self.scalers.keys())
        }