"""
CORE FRAUD DETECTION ENGINE
Complete implementation with ML + Rules
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import json
import warnings
warnings.filterwarnings('ignore')


class Config:
    """System Configuration"""
    CONTAMINATION = 0.05
    RISK_THRESHOLDS = {
        'CRITICAL': 0.85,
        'HIGH': 0.60,
        'MEDIUM': 0.30,
        'LOW': 0.00
    }


class DataGenerator:
    """Generate realistic synthetic data"""
    
    @staticmethod
    def generate_trips(n_trips=1000, fraud_rate=0.05, start_date=None):
        np.random.seed(42)
        
        if start_date is None:
            start_date = datetime.now() - timedelta(days=30)
            
        n_fraud = int(n_trips * fraud_rate)
        n_normal = n_trips - n_fraud
        
        conductors = ['Tendai M', 'John B', 'Sarah K', 'Mike R', 'Lisa T', 
                     'David N', 'Grace P', 'Paul S', 'Anna C', 'Robert K']
        routes = ['Harare-Bulawayo', 'Harare-Mutare', 'City Center-Ruwa', 
                 'CBD-Glen View', 'Airport-City', 'Downtown-Chitungwiza']
        
        # Normal trips
        normal = pd.DataFrame({
            'trip_id': [f'TRP-{i:06d}' for i in range(n_normal)],
            'trip_date': [start_date + timedelta(hours=np.random.randint(0, 720)) 
                         for _ in range(n_normal)],
            'conductor_id': np.random.choice([f'COND-{i:03d}' for i in range(1, 11)], n_normal),
            'conductor_name': np.random.choice(conductors, n_normal),
            'route_id': np.random.choice([f'R{i:03d}' for i in range(1, 7)], n_normal),
            'route_name': np.random.choice(routes, n_normal),
            'bus_id': np.random.choice([f'BUS-{i:03d}' for i in range(1, 26)], n_normal),
            'tickets_issued': np.random.poisson(45, n_normal),
            'ticket_fare': np.random.choice([3.00, 5.00, 7.00, 10.00], n_normal),
            'expenses_fuel': np.random.normal(80, 15, n_normal),
            'expenses_tolls': np.random.normal(20, 5, n_normal),
            'expenses_other': np.random.normal(10, 3, n_normal),
            'distance_km': np.random.normal(25, 8, n_normal),
            'trip_duration_min': np.random.normal(45, 15, n_normal),
            'weather': np.random.choice(['Sunny', 'Rainy', 'Cloudy'], n_normal),
            'day_of_week': np.random.choice(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'], n_normal),
            'is_fraud': 0
        })
        
        # Calculate revenue
        normal['electronic_revenue'] = normal['tickets_issued'] * normal['ticket_fare'] * 0.3
        normal['cash_collected'] = normal['tickets_issued'] * normal['ticket_fare'] * 0.7 + np.random.normal(0, 10, n_normal)
        
        # Fraud trips - various patterns
        fraud = pd.DataFrame({
            'trip_id': [f'TRP-{i+n_normal:06d}' for i in range(n_fraud)],
            'trip_date': [start_date + timedelta(hours=np.random.randint(0, 720)) 
                         for _ in range(n_fraud)],
            'conductor_id': np.random.choice([f'COND-{i:03d}' for i in range(1, 4)], n_fraud),
            'conductor_name': np.random.choice(conductors[:3], n_fraud),
            'route_id': np.random.choice([f'R{i:03d}' for i in range(1, 4)], n_fraud),
            'route_name': np.random.choice(routes[:3], n_fraud),
            'bus_id': np.random.choice([f'BUS-{i:03d}' for i in range(1, 11)], n_fraud),
            'tickets_issued': np.random.poisson(50, n_fraud),
            'ticket_fare': np.random.choice([3.00, 5.00, 7.00, 10.00], n_fraud),
            'expenses_fuel': np.random.normal(150, 30, n_fraud),
            'expenses_tolls': np.random.normal(40, 10, n_fraud),
            'expenses_other': np.random.normal(25, 8, n_fraud),
            'distance_km': np.random.normal(25, 8, n_fraud),
            'trip_duration_min': np.random.normal(45, 15, n_fraud),
            'weather': np.random.choice(['Sunny', 'Rainy', 'Cloudy'], n_fraud),
            'day_of_week': np.random.choice(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'], n_fraud),
            'is_fraud': 1
        })
        
        # Fraud revenue (skimming)
        fraud['electronic_revenue'] = fraud['tickets_issued'] * fraud['ticket_fare'] * 0.3
        fraud['cash_collected'] = fraud['tickets_issued'] * fraud['ticket_fare'] * 0.4
        
        df = pd.concat([normal, fraud], ignore_index=True)
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        return df


class FeatureEngineer:
    """Create fraud detection features"""
    
    def engineer_features(self, df):
        features = df.copy()
        
        # Revenue features
        features['expected_total_revenue'] = features['tickets_issued'] * features['ticket_fare']
        features['actual_total_revenue'] = features['electronic_revenue'] + features['cash_collected']
        features['revenue_gap_pct'] = (
            (features['expected_total_revenue'] - features['actual_total_revenue']) / 
            features['expected_total_revenue'].replace(0, 1)
        )
        features['cash_skimming_flag'] = (features['revenue_gap_pct'] > 0.15).astype(int)
        
        # Expense features
        features['total_expenses'] = (features['expenses_fuel'] + 
                                     features['expenses_tolls'] + 
                                     features['expenses_other'])
        features['expense_ratio'] = (
            features['total_expenses'] / 
            features['actual_total_revenue'].replace(0, 1)
        )
        features['fuel_per_km'] = features['expenses_fuel'] / features['distance_km'].replace(0, 1)
        features['toll_per_km'] = features['expenses_tolls'] / features['distance_km'].replace(0, 1)
        
        # Efficiency
        features['revenue_per_km'] = features['actual_total_revenue'] / features['distance_km'].replace(0, 1)
        features['revenue_per_minute'] = features['actual_total_revenue'] / features['trip_duration_min'].replace(0, 1)
        
        # Time features
        features['trip_hour'] = pd.to_datetime(features['trip_date']).dt.hour
        features['is_weekend'] = features['day_of_week'].isin(['Sat', 'Sun']).astype(int)
        features['is_night'] = ((features['trip_hour'] < 6) | (features['trip_hour'] > 22)).astype(int)
        features['is_rush_hour'] = features['trip_hour'].isin([7, 8, 17, 18]).astype(int)
        
        # Pattern features
        features['fuel_rounded'] = (features['expenses_fuel'] % 10 == 0).astype(int)
        features['toll_rounded'] = (features['expenses_tolls'] % 5 == 0).astype(int)
        features['both_rounded'] = (features['fuel_rounded'] & features['toll_rounded']).astype(int)
        
        return features
    
    def get_feature_columns(self):
        return [
            'revenue_gap_pct', 'expense_ratio', 'fuel_per_km', 'toll_per_km',
            'revenue_per_km', 'revenue_per_minute', 'is_weekend', 'is_night',
            'is_rush_hour', 'both_rounded', 'cash_skimming_flag'
        ]


class FraudDetectionModel:
    """ML Ensemble"""
    
    def __init__(self):
        self.isolation_forest = IsolationForest(
            n_estimators=100, contamination=Config.CONTAMINATION, random_state=42
        )
        self.random_forest = RandomForestClassifier(
            n_estimators=100, max_depth=10, class_weight='balanced', random_state=42
        )
        self.scaler = StandardScaler()
        self.is_trained = False
        
    def train(self, X, y=None):
        X_scaled = self.scaler.fit_transform(X)
        self.isolation_forest.fit(X_scaled)
        if y is not None:
            self.random_forest.fit(X_scaled, y)
        self.is_trained = True
        return self
    
    def predict(self, X):
        if not self.is_trained:
            raise ValueError("Model not trained!")
            
        X_scaled = self.scaler.transform(X)
        
        iso_pred = self.isolation_forest.predict(X_scaled)
        iso_scores = -self.isolation_forest.decision_function(X_scaled)
        iso_scores_norm = (iso_scores - iso_scores.min()) / (iso_scores.max() - iso_scores.min() + 1e-10)
        
        try:
            rf_proba = self.random_forest.predict_proba(X_scaled)[:, 1]
        except:
            rf_proba = np.zeros(len(X))
            
        ensemble_score = 0.6 * iso_scores_norm + 0.4 * rf_proba
        
        return {
            'is_anomaly': (iso_pred == -1).astype(int),
            'anomaly_score': iso_scores_norm,
            'fraud_probability': ensemble_score,
            'ensemble_score': ensemble_score
        }


class RuleEngine:
    """Business Rules"""
    
    RULES = [
        {'id': 'R001', 'name': 'Severe Revenue Shortfall', 
         'check': lambda r: r['revenue_gap_pct'] > 0.20, 'severity': 'CRITICAL', 'weight': 0.35},
        {'id': 'R002', 'name': 'Excessive Expense Ratio', 
         'check': lambda r: r['expense_ratio'] > 0.60, 'severity': 'HIGH', 'weight': 0.25},
        {'id': 'R003', 'name': 'Suspicious Fuel Efficiency', 
         'check': lambda r: r['fuel_per_km'] > 8, 'severity': 'HIGH', 'weight': 0.20},
        {'id': 'R004', 'name': 'Rounded Expense Pattern', 
         'check': lambda r: r['both_rounded'] == 1, 'severity': 'MEDIUM', 'weight': 0.15},
        {'id': 'R005', 'name': 'Night Weekend Trip', 
         'check': lambda r: r['is_night'] == 1 and r['is_weekend'] == 1, 'severity': 'MEDIUM', 'weight': 0.05},
    ]
    
    def apply_rules(self, row):
        triggered = []
        total_weight = 0
        
        for rule in self.RULES:
            if rule['check'](row):
                triggered.append({
                    'rule_id': rule['id'],
                    'rule_name': rule['name'],
                    'severity': rule['severity']
                })
                total_weight += rule['weight']
                
        severities = [t['severity'] for t in triggered]
        max_sev = 'LOW'
        for s in ['CRITICAL', 'HIGH', 'MEDIUM']:
            if s in severities:
                max_sev = s
                break
                
        return {
            'triggered_rules': triggered,
            'rule_score': min(total_weight, 1.0),
            'max_severity': max_sev
        }


class RiskScorer:
    """Calculate risk scores"""
    
    def calculate_risk(self, ml_result, rule_result):
        ml_score = ml_result['ensemble_score']
        rule_score = rule_result['rule_score']
        final_score = 0.6 * ml_score + 0.4 * rule_score
        
        thresholds = Config.RISK_THRESHOLDS
        if final_score >= thresholds['CRITICAL'] or rule_result['max_severity'] == 'CRITICAL':
            category = 'CRITICAL'
        elif final_score >= thresholds['HIGH']:
            category = 'HIGH'
        elif final_score >= thresholds['MEDIUM']:
            category = 'MEDIUM'
        else:
            category = 'LOW'
            
        confidence = abs(ml_score - 0.5) * 2
        
        return {
            'risk_score': round(final_score, 3),
            'risk_category': category,
            'ml_component': round(ml_score, 3),
            'rule_component': round(rule_score, 3),
            'confidence': round(confidence, 3)
        }


class FraudDetectionSystem:
    """Main System"""
    
    def __init__(self):
        self.data_generator = DataGenerator()
        self.feature_engineer = FeatureEngineer()
        self.ml_model = FraudDetectionModel()
        self.rule_engine = RuleEngine()
        self.risk_scorer = RiskScorer()
        self.alerts = []
        self.current_data = None
        self.current_results = None
        
    def generate_data(self, n_trips=1000, fraud_rate=0.05):
        self.current_data = self.data_generator.generate_trips(n_trips, fraud_rate)
        return self.current_data
    
    def analyze(self, df=None):
        if df is None:
            df = self.current_data
            
        if df is None:
            raise ValueError("No data available!")
        
        # Features
        features_df = self.feature_engineer.engineer_features(df)
        feature_cols = self.feature_engineer.get_feature_columns()
        X = features_df[feature_cols].fillna(0)
        
        # Train if needed
        if not self.ml_model.is_trained:
            y = features_df['is_fraud'] if 'is_fraud' in features_df.columns else None
            self.ml_model.train(X, y)
        
        # Predict
        ml_results = self.ml_model.predict(X)
        
        # Process each row
        results = []
        self.alerts = []
        
        for idx, row in features_df.iterrows():
            rule_result = self.rule_engine.apply_rules(row)
            risk = self.risk_scorer.calculate_risk(
                {k: v[idx] for k, v in ml_results.items()}, rule_result
            )
            
            # Create alert
            if risk['risk_category'] in ['HIGH', 'CRITICAL']:
                self.alerts.append({
                    'alert_id': f"ALT-{datetime.now().strftime('%Y%m%d%H%M%S')}-{idx}",
                    'timestamp': datetime.now().isoformat(),
                    'trip_id': row['trip_id'],
                    'conductor': row['conductor_name'],
                    'route': row['route_name'],
                    'risk_score': risk['risk_score'],
                    'category': risk['risk_category'],
                    'factors': [r['rule_name'] for r in rule_result['triggered_rules']],
                    'action': self._get_action(risk['risk_category'])
                })
            
            results.append({
                'trip_id': row['trip_id'],
                'conductor_name': row['conductor_name'],
                'conductor_id': row['conductor_id'],
                'route_name': row['route_name'],
                'trip_date': row['trip_date'].strftime('%Y-%m-%d %H:%M') if isinstance(row['trip_date'], datetime) else str(row['trip_date']),
                'actual_fraud': int(row.get('is_fraud', 0)),
                'risk_score': risk['risk_score'],
                'risk_category': risk['risk_category'],
                'ml_score': risk['ml_component'],
                'rule_score': risk['rule_component'],
                'confidence': risk['confidence'],
                'revenue_gap_pct': round(row['revenue_gap_pct'], 3),
                'expense_ratio': round(row['expense_ratio'], 3),
                'fuel_per_km': round(row['fuel_per_km'], 2),
                'triggered_rules': json.dumps([r['rule_name'] for r in rule_result['triggered_rules']])
            })
        
        self.current_results = pd.DataFrame(results)
        return self.current_results
    
    def _get_action(self, category):
        actions = {
            'CRITICAL': 'SUSPEND: Immediate investigation required',
            'HIGH': 'INSPECT: Review within 24 hours',
            'MEDIUM': 'MONITOR: Include in next audit',
            'LOW': 'STANDARD: Normal processing'
        }
        return actions.get(category, 'Review')
    
    def get_summary(self):
        if self.current_results is None:
            return None
            
        r = self.current_results
        return {
            'total': len(r),
            'critical': len(r[r['risk_category'] == 'CRITICAL']),
            'high': len(r[r['risk_category'] == 'HIGH']),
            'medium': len(r[r['risk_category'] == 'MEDIUM']),
            'low': len(r[r['risk_category'] == 'LOW']),
            'alerts': len(self.alerts),
            'detection_rate': len(r[(r['actual_fraud']==1) & (r['risk_category'].isin(['HIGH','CRITICAL']))]) / r['actual_fraud'].sum() if r['actual_fraud'].sum() > 0 else 0,
            'top_conductors': r.groupby('conductor_name')['risk_score'].mean().nlargest(5).to_dict()
        }
    
    def export_results(self, filename=None):
        if self.current_results is None:
            return None
            
        if filename is None:
            filename = f"exports/fraud_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        
        self.current_results.to_excel(filename, index=False, engine='openpyxl')
        return filename


# Global system instance
_system = None

def get_system():
    global _system
    if _system is None:
        _system = FraudDetectionSystem()
    return _system