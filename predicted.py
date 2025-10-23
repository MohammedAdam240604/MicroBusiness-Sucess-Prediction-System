import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, roc_curve
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

class BusinessSuccessPredictor:
    def __init__(self, csv_file):
        """Initialize the predictor with CSV file"""
        self.csv_file = csv_file
        self.df = None
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = []
        self.label_encoder = LabelEncoder()
        
    def preprocess_data(self):
        """Clean and prepare data for analysis"""
        print("📊 Preprocessing data...")
        
        # Load data if not already loaded
        if self.df is None:
            self.df = pd.read_csv(self.csv_file, sep=',')
        
        print(f"Dataset shape: {self.df.shape}")
        print(f"Columns: {list(self.df.columns)}")
        
        # Handle missing business names
        self.df['name'].fillna('Unknown', inplace=True)
        
        # Convert numeric columns
        numeric_columns = ['Monthly_Revenue', 'Customer_Count', 'Growth_Rate', 
                          'Digital_Marketing_Reach', 'Customer_Engagement', 
                          'Funding_Available', 'Operating_Months', 'Team_Size',
                          'Innovation_Score', 'Sustainability_Index', 
                          'Founder_Experience', 'Local_Economy_Index', 'Social_Sentiment',
                          'latitude', 'longitude', 'success_score']
        
        for col in numeric_columns:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
        
        # Encode Competition Level
        if 'Competition_Level' in self.df.columns:
            competition_map = {'Low': 1, 'Medium': 2, 'High': 3}
            self.df['Competition_Level_Encoded'] = self.df['Competition_Level'].map(competition_map)
            self.df['Competition_Level_Encoded'].fillna(2, inplace=True)
        
        # Use the existing success_label column (0 = failure, 1 = success)
        if 'success_label' in self.df.columns:
            self.df['success_label'] = self.df['success_label'].astype(int)
            print(f"\n✅ Success distribution:")
            print(f"   Successful (1): {(self.df['success_label'] == 1).sum()} ({(self.df['success_label'] == 1).sum()/len(self.df)*100:.1f}%)")
            print(f"   Unsuccessful (0): {(self.df['success_label'] == 0).sum()} ({(self.df['success_label'] == 0).sum()/len(self.df)*100:.1f}%)")
        
        print(f"\n✅ Data loaded: {len(self.df)} businesses")
        print(f"   Missing values: {self.df.isnull().sum().sum()}")
        return self
    
    def feature_engineering(self):
        """Create additional features for better predictions"""
        print("\n🔧 Engineering features...")
        
        # Revenue per customer
        self.df['revenue_per_customer'] = self.df['Monthly_Revenue'] / (self.df['Customer_Count'] + 1)
        
        # Efficiency score (revenue relative to team size)
        self.df['team_efficiency'] = self.df['Monthly_Revenue'] / (self.df['Team_Size'] + 1)
        
        # Marketing efficiency (revenue per marketing reach)
        self.df['marketing_efficiency'] = self.df['Monthly_Revenue'] / (self.df['Digital_Marketing_Reach'] + 1)
        
        # Experience-to-team ratio
        self.df['experience_ratio'] = self.df['Founder_Experience'] / (self.df['Team_Size'] + 1)
        
        # Funding utilization (revenue per funding available)
        self.df['funding_efficiency'] = self.df['Monthly_Revenue'] / (self.df['Funding_Available'] + 1)
        
        # Sustainability-Innovation index
        self.df['sustain_innovation'] = (self.df['Sustainability_Index'] + self.df['Innovation_Score']) / 2
        
        # Business maturity factor
        self.df['maturity_factor'] = np.log1p(self.df['Operating_Months'])
        
        print("✅ Created 7 engineered features")
        return self
    
    def train_prediction_model(self, test_size=0.2):
        """Train model to predict business success (0 or 1)"""
        print("\n🤖 Training ML prediction model...")
        
        # Select features - using all relevant columns
        potential_features = [
            'Monthly_Revenue', 'Customer_Count', 'Growth_Rate', 
            'Digital_Marketing_Reach', 'Customer_Engagement',
            'Funding_Available', 'Operating_Months', 'Team_Size',
            'Innovation_Score', 'Sustainability_Index', 
            'Founder_Experience', 'Local_Economy_Index', 'Social_Sentiment',
            'Competition_Level_Encoded',
            # Engineered features
            'revenue_per_customer', 'team_efficiency', 'marketing_efficiency',
            'experience_ratio', 'funding_efficiency', 'sustain_innovation',
            'maturity_factor'
        ]
        
        # Only use features that exist in dataset
        self.feature_columns = [col for col in potential_features if col in self.df.columns]
        print(f"   Using {len(self.feature_columns)} features")
        
        # Prepare data
        X = self.df[self.feature_columns].fillna(self.df[self.feature_columns].median())
        y = self.df['success_label']
        
        # Remove any remaining NaN in target
        valid_indices = ~y.isna()
        X = X[valid_indices]
        y = y[valid_indices]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train Random Forest Classifier
        self.model = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            class_weight='balanced'  # Handle imbalanced data
        )
        self.model.fit(X_train_scaled, y_train)
        
        # Predictions
        y_pred = self.model.predict(X_test_scaled)
        y_pred_proba = self.model.predict_proba(X_test_scaled)[:, 1]
        
        # Evaluation metrics
        accuracy = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        print(f"\n✅ Model Performance:")
        print(f"   Accuracy: {accuracy*100:.2f}%")
        print(f"   ROC-AUC Score: {roc_auc:.3f}")
        print(f"   Training samples: {len(X_train)}")
        print(f"   Test samples: {len(X_test)}")
        
        # Classification report
        print("\n📊 Detailed Classification Report:")
        print(classification_report(y_test, y_pred, 
                                   target_names=['Unsuccessful (0)', 'Successful (1)']))
        
        # Store test data for visualization
        self.X_test = X_test_scaled
        self.y_test = y_test
        self.y_pred = y_pred
        self.y_pred_proba = y_pred_proba
        
        return self
    
    def save_model(self, model_path='trained_model.pkl', scaler_path='scaler.pkl', features_path='feature_columns.pkl'):
        """Save the trained model, scaler, and feature columns"""
        if self.model is None:
            print("❌ No model to save! Train the model first.")
            return False
        
        try:
            # Save model
            with open(model_path, 'wb') as f:
                pickle.dump(self.model, f)
            
            # Save scaler
            with open(scaler_path, 'wb') as f:
                pickle.dump(self.scaler, f)
            
            # Save feature columns
            with open(features_path, 'wb') as f:
                pickle.dump(self.feature_columns, f)
            
            print(f"\n✅ Model saved successfully!")
            print(f"   📁 Model: {model_path}")
            print(f"   📁 Scaler: {scaler_path}")
            print(f"   📁 Features: {features_path}")
            return True
        
        except Exception as e:
            print(f"❌ Error saving model: {e}")
            return False
    
    def load_model(self, model_path='trained_model.pkl', scaler_path='scaler.pkl', features_path='feature_columns.pkl'):
        """Load a previously trained model"""
        try:
            # Check if all files exist
            if not all(os.path.exists(f) for f in [model_path, scaler_path, features_path]):
                print(f"❌ Model files not found. Please train the model first.")
                return False
            
            # Load model
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            
            # Load scaler
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            
            # Load feature columns
            with open(features_path, 'rb') as f:
                self.feature_columns = pickle.load(f)
            
            print(f"\n✅ Model loaded successfully!")
            print(f"   Using {len(self.feature_columns)} features")
            return True
        
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    
    def predict_new_business(self, business_data):
        """
        Predict success for a new business
        
        Parameters:
        -----------
        business_data : dict
            Dictionary with business features. Example:
            {
                'Monthly_Revenue': 35000,
                'Customer_Count': 250,
                'Growth_Rate': 15.5,
                'Digital_Marketing_Reach': 12000,
                'Customer_Engagement': 7,
                'Funding_Available': 50000,
                'Operating_Months': 24,
                'Team_Size': 8,
                'Innovation_Score': 7,
                'Sustainability_Index': 6,
                'Founder_Experience': 10,
                'Local_Economy_Index': 7,
                'Social_Sentiment': 8,
                'Competition_Level': 'Medium'  # or 'Low' or 'High'
            }
        
        Returns:
        --------
        dict with prediction results
        """
        if self.model is None:
            return {'error': 'Model not trained yet! Call train_prediction_model() first.'}
        
        # Encode competition level
        if 'Competition_Level' in business_data:
            comp_map = {'Low': 1, 'Medium': 2, 'High': 3}
            business_data['Competition_Level_Encoded'] = comp_map.get(business_data['Competition_Level'], 2)
        
        # Calculate engineered features
        business_data['revenue_per_customer'] = business_data.get('Monthly_Revenue', 0) / (business_data.get('Customer_Count', 1) + 1)
        business_data['team_efficiency'] = business_data.get('Monthly_Revenue', 0) / (business_data.get('Team_Size', 1) + 1)
        business_data['marketing_efficiency'] = business_data.get('Monthly_Revenue', 0) / (business_data.get('Digital_Marketing_Reach', 1) + 1)
        business_data['experience_ratio'] = business_data.get('Founder_Experience', 0) / (business_data.get('Team_Size', 1) + 1)
        business_data['funding_efficiency'] = business_data.get('Monthly_Revenue', 0) / (business_data.get('Funding_Available', 1) + 1)
        business_data['sustain_innovation'] = (business_data.get('Sustainability_Index', 0) + business_data.get('Innovation_Score', 0)) / 2
        business_data['maturity_factor'] = np.log1p(business_data.get('Operating_Months', 1))
        
        # Extract features in correct order
        features = [business_data.get(col, 0) for col in self.feature_columns]
        features_scaled = self.scaler.transform([features])
        
        # Make prediction
        prediction = self.model.predict(features_scaled)[0]
        probability = self.model.predict_proba(features_scaled)[0]
        
        # Get feature importance for this prediction
        feature_impact = dict(zip(self.feature_columns, features))
        
        result = {
            'prediction': 'Successful ✅' if prediction == 1 else 'Unsuccessful ❌',
            'success_probability': probability[1] * 100,
            'failure_probability': probability[0] * 100,
            'confidence': max(probability) * 100,
            'recommendation': self._generate_recommendation(prediction, probability[1], business_data)
        }
        
        return result
    
    def _generate_recommendation(self, prediction, success_prob, business_data):
        """Generate actionable recommendations"""
        recommendations = []
        
        if prediction == 1 and success_prob > 0.7:
            recommendations.append("✅ Strong success indicators! Focus on scaling.")
            recommendations.append("💡 Consider expanding digital marketing reach")
            recommendations.append("📈 Monitor growth metrics closely")
        elif prediction == 1 and success_prob >= 0.5:
            recommendations.append("⚠️ Moderate success probability. Strengthen key areas:")
            if business_data.get('Customer_Engagement', 0) < 6:
                recommendations.append("  • Improve customer engagement strategies")
            if business_data.get('Digital_Marketing_Reach', 0) < 10000:
                recommendations.append("  • Increase digital marketing efforts")
            if business_data.get('Growth_Rate', 0) < 12:
                recommendations.append("  • Focus on accelerating growth rate")
        else:
            recommendations.append("❌ High risk of failure. Critical improvements needed:")
            if business_data.get('Monthly_Revenue', 0) < 20000:
                recommendations.append("  • Urgent: Improve revenue streams")
            if business_data.get('Customer_Engagement', 0) < 5:
                recommendations.append("  • Critical: Enhance customer engagement")
            if business_data.get('Innovation_Score', 0) < 5:
                recommendations.append("  • Important: Increase innovation initiatives")
            recommendations.append("💡 Consider pivoting business model or seeking mentorship")
        
        return '\n'.join(recommendations)
    
    def get_success_factors(self):
        """Analyze what makes businesses successful"""
        successful = self.df[self.df['success_label'] == 1]
        unsuccessful = self.df[self.df['success_label'] == 0]
        
        print("\n📊 Success Factors Analysis:")
        print("="*60)
        
        comparison_features = [
            'Monthly_Revenue', 'Growth_Rate', 'Customer_Engagement',
            'Innovation_Score', 'Digital_Marketing_Reach', 'Customer_Count'
        ]
        
        for feature in comparison_features:
            if feature in self.df.columns:
                succ_mean = successful[feature].mean()
                unsucc_mean = unsuccessful[feature].mean()
                diff_pct = ((succ_mean - unsucc_mean) / unsucc_mean * 100) if unsucc_mean != 0 else 0
                
                print(f"\n{feature}:")
                print(f"  Successful:   {succ_mean:,.2f}")
                print(f"  Unsuccessful: {unsucc_mean:,.2f}")
                print(f"  Difference:   {diff_pct:+.1f}%")
    
    def visualize_results(self):
        """Create comprehensive visualizations"""
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. Success Distribution
        ax1 = fig.add_subplot(gs[0, 0])
        success_counts = self.df['success_label'].value_counts()
        colors = ['#ff6b6b', '#51cf66']
        ax1.pie(success_counts, labels=['Unsuccessful', 'Successful'], autopct='%1.1f%%',
                colors=colors, startangle=90)
        ax1.set_title('Success Distribution', fontsize=14, fontweight='bold')
        
        # 2. Confusion Matrix
        ax2 = fig.add_subplot(gs[0, 1])
        cm = confusion_matrix(self.y_test, self.y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax2)
        ax2.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Actual')
        ax2.set_xlabel('Predicted')
        
        # 3. ROC Curve
        ax3 = fig.add_subplot(gs[0, 2])
        fpr, tpr, _ = roc_curve(self.y_test, self.y_pred_proba)
        roc_auc = roc_auc_score(self.y_test, self.y_pred_proba)
        ax3.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        ax3.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax3.set_xlabel('False Positive Rate')
        ax3.set_ylabel('True Positive Rate')
        ax3.set_title('ROC Curve', fontsize=14, fontweight='bold')
        ax3.legend(loc="lower right")
        
        # 4. Feature Importance
        ax4 = fig.add_subplot(gs[1, :])
        importance_df = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False).head(15)
        sns.barplot(data=importance_df, x='importance', y='feature', palette='viridis', ax=ax4)
        ax4.set_title('Top 15 Feature Importance', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Importance Score')
        
        # 5. Revenue vs Growth (colored by success)
        ax5 = fig.add_subplot(gs[2, 0])
        scatter = ax5.scatter(self.df['Monthly_Revenue'], self.df['Growth_Rate'],
                            c=self.df['success_label'], cmap='RdYlGn', alpha=0.6, s=50)
        ax5.set_xlabel('Monthly Revenue (₹)')
        ax5.set_ylabel('Growth Rate (%)')
        ax5.set_title('Revenue vs Growth Rate', fontsize=14, fontweight='bold')
        plt.colorbar(scatter, ax=ax5, label='Success (0=No, 1=Yes)')
        
        # 6. Engagement vs Innovation
        ax6 = fig.add_subplot(gs[2, 1])
        scatter2 = ax6.scatter(self.df['Customer_Engagement'], self.df['Innovation_Score'],
                             c=self.df['success_label'], cmap='RdYlGn', alpha=0.6, s=50)
        ax6.set_xlabel('Customer Engagement')
        ax6.set_ylabel('Innovation Score')
        ax6.set_title('Engagement vs Innovation', fontsize=14, fontweight='bold')
        plt.colorbar(scatter2, ax=ax6, label='Success (0=No, 1=Yes)')
        
        # 7. Business Type Success Rate
        ax7 = fig.add_subplot(gs[2, 2])
        if 'type' in self.df.columns:
            type_success = self.df.groupby('type')['success_label'].agg(['mean', 'count'])
            type_success = type_success[type_success['count'] >= 5].sort_values('mean', ascending=False).head(10)
            type_success['mean'].plot(kind='barh', ax=ax7, color='skyblue')
            ax7.set_xlabel('Success Rate')
            ax7.set_title('Top 10 Business Types by Success Rate', fontsize=14, fontweight='bold')
        
        plt.savefig('prediction_results.png', dpi=300, bbox_inches='tight')
        print("\n📊 Visualizations saved as 'prediction_results.png'")
        plt.show()
    
    def generate_full_report(self, output_file='prediction_report.txt'):
        """Generate comprehensive report"""
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("MICRO BUSINESS SUCCESS PREDICTION REPORT\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"Total Businesses: {len(self.df)}\n")
            f.write(f"Successful: {(self.df['success_label'] == 1).sum()} ({(self.df['success_label'] == 1).sum()/len(self.df)*100:.1f}%)\n")
            f.write(f"Unsuccessful: {(self.df['success_label'] == 0).sum()} ({(self.df['success_label'] == 0).sum()/len(self.df)*100:.1f}%)\n\n")
            
            # Success factors
            f.write("-"*80 + "\n")
            f.write("KEY SUCCESS FACTORS\n")
            f.write("-"*80 + "\n")
            
            successful = self.df[self.df['success_label'] == 1]
            unsuccessful = self.df[self.df['success_label'] == 0]
            
            metrics = ['Monthly_Revenue', 'Growth_Rate', 'Customer_Engagement', 
                      'Innovation_Score', 'Digital_Marketing_Reach']
            
            for metric in metrics:
                if metric in self.df.columns:
                    succ_avg = successful[metric].mean()
                    unsucc_avg = unsuccessful[metric].mean()
                    f.write(f"\n{metric}:\n")
                    f.write(f"  Successful:   {succ_avg:,.2f}\n")
                    f.write(f"  Unsuccessful: {unsucc_avg:,.2f}\n")
                    if unsucc_avg != 0:
                        diff = ((succ_avg - unsucc_avg) / unsucc_avg * 100)
                        f.write(f"  Difference:   {diff:+.1f}%\n")
            
            # Top performers
            f.write("\n" + "-"*80 + "\n")
            f.write("TOP 10 SUCCESSFUL BUSINESSES\n")
            f.write("-"*80 + "\n")
            top_successful = self.df[self.df['success_label'] == 1].nlargest(10, 'success_score')
            cols = ['name', 'type', 'Monthly_Revenue', 'Growth_Rate', 'success_score']
            existing_cols = [c for c in cols if c in top_successful.columns]
            f.write(top_successful[existing_cols].to_string(index=False))
        
        print(f"\n📝 Full report saved as '{output_file}'")


# ============================================================================
# MAIN EXECUTION
# ============================================================================
if __name__ == "__main__":
    print("\n" + "="*80)
    print("🚀 MICRO BUSINESS SUCCESS PREDICTION SYSTEM")
    print("="*80 + "\n")
    
    # Initialize
    predictor = BusinessSuccessPredictor('coimbatore_enriched_labeled.csv')
    
    # Check if trained model exists
    model_exists = os.path.exists('trained_model.pkl')
    
    if model_exists:
        print("📦 Found existing trained model!")
        print("   Loading saved model (fast)...\n")
        
        # Load the saved model
        if predictor.load_model():
            # Still need to load data for full analysis/visualization
            predictor.preprocess_data()
            predictor.feature_engineering()
            
            print("\n💡 Model loaded successfully! Skipping training.")
            print("   To retrain, delete 'trained_model.pkl' and run again.\n")
        else:
            print("\n⚠️ Failed to load model. Training new model...\n")
            model_exists = False
    
    if not model_exists:
        print("🤖 No saved model found. Training new model...\n")
        
        # Process data
        predictor.preprocess_data()
        predictor.feature_engineering()
        
        # Train model
        predictor.train_prediction_model(test_size=0.2)
        
        # Save the trained model for future use
        predictor.save_model()
        
        # Analyze success factors
        predictor.get_success_factors()
        
        # Visualize
        print("\n📊 Generating visualizations...")
        predictor.visualize_results()
        
        # Generate report
        predictor.generate_full_report()
    
    # Example: Predict a new business
    print("\n" + "="*80)
    print("🔮 EXAMPLE: Predicting a New Business")
    print("="*80)
    
    new_business = {
        'Monthly_Revenue': 12000,
        'Customer_Count': 250,
        'Growth_Rate': 18.5,
        'Digital_Marketing_Reach': 15000,
        'Customer_Engagement': 8,
        'Funding_Available': 10000,
        'Operating_Months': 24,
        'Team_Size': 10,
        'Innovation_Score': 7,
        'Sustainability_Index': 6,
        'Founder_Experience': 12,
        'Local_Economy_Index': 7,
        'Social_Sentiment': 8,
        'Competition_Level': 'Medium'
    }
    
    result = predictor.predict_new_business(new_business)
    
    print(f"\n📊 Business Details:")
    print(f"   Revenue: ₹{new_business['Monthly_Revenue']:,}")
    print(f"   Customers: {new_business['Customer_Count']}")
    print(f"   Growth Rate: {new_business['Growth_Rate']}%")
    print(f"   Team Size: {new_business['Team_Size']}")
    
    print(f"\n🎯 Prediction: {result['prediction']}")
    print(f"   Success Probability: {result['success_probability']:.1f}%")
    print(f"   Failure Probability: {result['failure_probability']:.1f}%")
    print(f"   Confidence: {result['confidence']:.1f}%")
    print(f"\n💡 Recommendations:\n{result['recommendation']}")
    
    print("\n" + "="*80)
    print("✅ Analysis Complete!")
    print("="*80 + "\n")