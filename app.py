from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import pickle
import os

app = Flask(__name__)
CORS(app)

# Global variables
predictor_model = None
scaler = None
feature_columns = None

def load_trained_model():
    """Load the trained model, scaler, and feature columns"""
    global predictor_model, scaler, feature_columns
    
    try:
        if not all(os.path.exists(f) for f in ['trained_model.pkl', 'scaler.pkl', 'feature_columns.pkl']):
            print("❌ Model files not found!")
            print("Please run the training script first to generate:")
            print("   - trained_model.pkl")
            print("   - scaler.pkl")
            print("   - feature_columns.pkl")
            return False
        
        with open('trained_model.pkl', 'rb') as f:
            predictor_model = pickle.load(f)
        
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        
        with open('feature_columns.pkl', 'rb') as f:
            feature_columns = pickle.load(f)
        
        print("✅ Model loaded successfully!")
        print(f"   Features: {len(feature_columns)}")
        return True
    
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False


def map_html_to_model_format(html_data):
    """Convert HTML camelCase field names to model's Snake_Case format"""
    return {
        'Monthly_Revenue': html_data.get('monthlyRevenue', 0),
        'Customer_Count': html_data.get('customerCount', 0),
        'Growth_Rate': html_data.get('growthRate', 0),
        'Digital_Marketing_Reach': html_data.get('marketingReach', 0),
        'Customer_Engagement': html_data.get('engagementScore', 0),
        'Funding_Available': html_data.get('fundingAvailable', 0),
        'Operating_Months': html_data.get('operatingMonths', 0),
        'Team_Size': html_data.get('teamSize', 0),
        'Innovation_Score': html_data.get('innovationScore', 0),
        'Sustainability_Index': html_data.get('sustainabilityIndex', 0),
        'Founder_Experience': html_data.get('founderExperience', 0),
        'Local_Economy_Index': html_data.get('economyIndex', 0),
        'Social_Sentiment': html_data.get('socialSentiment', 0),
        'Competition_Level': html_data.get('competitionLevel', 'Medium')
    }


def calculate_engineered_features(data):
    """Calculate engineered features exactly as training script does"""
    comp_map = {'Low': 1, 'Medium': 2, 'High': 3}
    competition_encoded = comp_map.get(data.get('Competition_Level', 'Medium'), 2)
    
    revenue = data.get('Monthly_Revenue', 0)
    customers = data.get('Customer_Count', 1)
    team = data.get('Team_Size', 1)
    marketing = data.get('Digital_Marketing_Reach', 1)
    funding = data.get('Funding_Available', 1)
    experience = data.get('Founder_Experience', 0)
    sustainability = data.get('Sustainability_Index', 0)
    innovation = data.get('Innovation_Score', 0)
    months = data.get('Operating_Months', 1)
    
    features = {
        'Monthly_Revenue': revenue,
        'Customer_Count': customers,
        'Growth_Rate': data.get('Growth_Rate', 0),
        'Digital_Marketing_Reach': marketing,
        'Customer_Engagement': data.get('Customer_Engagement', 0),
        'Funding_Available': funding,
        'Operating_Months': months,
        'Team_Size': team,
        'Innovation_Score': innovation,
        'Sustainability_Index': sustainability,
        'Founder_Experience': experience,
        'Local_Economy_Index': data.get('Local_Economy_Index', 0),
        'Social_Sentiment': data.get('Social_Sentiment', 0),
        'Competition_Level_Encoded': competition_encoded,
        'revenue_per_customer': revenue / (customers + 1),
        'team_efficiency': revenue / (team + 1),
        'marketing_efficiency': revenue / (marketing + 1),
        'experience_ratio': experience / (team + 1),
        'funding_efficiency': revenue / (funding + 1),
        'sustain_innovation': (sustainability + innovation) / 2,
        'maturity_factor': np.log1p(months)
    }
    
    return features


def calculate_component_scores(data):
    """Calculate individual component scores for visualization"""
    scores = {}
    
    revenue = data.get('Monthly_Revenue', 0)
    if revenue >= 50000:
        scores['revenue_score'] = 100
    elif revenue >= 30000:
        scores['revenue_score'] = 75
    elif revenue >= 15000:
        scores['revenue_score'] = 50
    else:
        scores['revenue_score'] = max(0, (revenue / 15000) * 50)
    
    growth = data.get('Growth_Rate', 0)
    if growth >= 20:
        scores['growth_score'] = 100
    elif growth >= 10:
        scores['growth_score'] = 70
    elif growth >= 0:
        scores['growth_score'] = 50
    else:
        scores['growth_score'] = max(0, 50 + growth * 2)
    
    engagement = data.get('Customer_Engagement', 0)
    scores['engagement_score'] = engagement * 10
    
    innovation = data.get('Innovation_Score', 0)
    scores['innovation_score'] = innovation * 10
    
    marketing = data.get('Digital_Marketing_Reach', 0)
    if marketing >= 15000:
        scores['marketing_score'] = 100
    elif marketing >= 8000:
        scores['marketing_score'] = 70
    elif marketing >= 3000:
        scores['marketing_score'] = 40
    else:
        scores['marketing_score'] = max(0, (marketing / 3000) * 40)
    
    return scores


def generate_insights(data, overall_score, ml_prediction):
    """Generate strengths, weaknesses, and recommendations"""
    strengths = []
    weaknesses = []
    recommendations = []
    
    revenue = data.get('Monthly_Revenue', 0)
    if revenue >= 30000:
        strengths.append(f"Strong revenue generation (₹{revenue:,}/month)")
    elif revenue < 15000:
        weaknesses.append("Revenue below sustainable threshold")
        recommendations.append("Focus on increasing pricing or customer acquisition")
    
    growth = data.get('Growth_Rate', 0)
    if growth >= 15:
        strengths.append(f"Excellent growth trajectory ({growth}% monthly)")
    elif growth < 5:
        weaknesses.append("Limited growth momentum")
        recommendations.append("Implement aggressive growth strategies and market expansion")
    
    engagement = data.get('Customer_Engagement', 0)
    if engagement >= 7:
        strengths.append(f"High customer engagement (Score: {engagement}/10)")
    elif engagement < 5:
        weaknesses.append("Low customer engagement levels")
        recommendations.append("Improve customer experience and retention programs")
    
    innovation = data.get('Innovation_Score', 0)
    if innovation >= 7:
        strengths.append(f"Strong innovation capability (Score: {innovation}/10)")
    elif innovation < 5:
        weaknesses.append("Limited innovation initiatives")
        recommendations.append("Invest in R&D and product differentiation")
    
    marketing = data.get('Digital_Marketing_Reach', 0)
    if marketing >= 10000:
        strengths.append(f"Wide digital marketing reach ({marketing:,} contacts)")
    elif marketing < 5000:
        weaknesses.append("Limited marketing reach")
        recommendations.append("Expand digital marketing channels and social media presence")
    
    team = data.get('Team_Size', 0)
    if team >= 8:
        strengths.append(f"Well-staffed team ({team} members)")
    elif team < 3:
        weaknesses.append("Understaffed team")
        recommendations.append("Consider strategic hiring to support growth")
    
    funding = data.get('Funding_Available', 0)
    if funding >= 75000:
        strengths.append(f"Strong funding position (₹{funding:,})")
    elif funding < 25000:
        weaknesses.append("Limited funding reserves")
        recommendations.append("Explore funding options: loans, investors, or grants")
    
    competition = data.get('Competition_Level', 'Medium')
    if competition == 'Low':
        strengths.append("Operating in low-competition market")
    elif competition == 'High':
        weaknesses.append("High competitive pressure")
        recommendations.append("Focus on unique value proposition and differentiation")
    
    if ml_prediction == 1 and overall_score >= 70:
        recommendations.insert(0, "ML Model predicts SUCCESS! Focus on scaling and expansion")
    elif ml_prediction == 1:
        recommendations.insert(0, "ML Model predicts SUCCESS with caution. Strengthen key areas")
    elif overall_score >= 50:
        recommendations.insert(0, "Mixed signals. Critical improvements needed to ensure success")
    else:
        recommendations.insert(0, "ML Model predicts CHALLENGES. Immediate action required")
    
    if len(strengths) < 3:
        strengths.append("Business is operational and serving customers")
        strengths.append("Foundation established for growth")
    if len(weaknesses) < 3:
        weaknesses.append("Monitor market conditions closely")
    if len(recommendations) < 3:
        recommendations.append("Regularly review and adjust strategy based on performance")
    
    return {
        'strengths': strengths[:5],
        'weaknesses': weaknesses[:5],
        'recommendations': recommendations[:5]
    }


@app.route('/')
def home():
    """Serve the HTML frontend"""
    html_content = open('frontend.html', 'r', encoding='utf-8').read()
    return html_content


@app.route('/api/predict', methods=['POST'])
def predict():
    """Main prediction endpoint"""
    try:
        if predictor_model is None:
            return jsonify({
                'success': False,
                'error': 'Model not loaded. Please ensure trained model files exist.'
            }), 500
        
        html_data = request.json
        
        required_fields = [
            'monthlyRevenue', 'customerCount', 'growthRate', 'marketingReach',
            'engagementScore', 'fundingAvailable', 'operatingMonths', 'teamSize',
            'innovationScore', 'sustainabilityIndex', 'competitionLevel',
            'founderExperience', 'economyIndex', 'socialSentiment'
        ]
        
        missing_fields = [f for f in required_fields if f not in html_data]
        if missing_fields:
            return jsonify({
                'success': False,
                'error': f'Missing required fields: {", ".join(missing_fields)}'
            }), 400
        
        model_data = map_html_to_model_format(html_data)
        features_dict = calculate_engineered_features(model_data)
        feature_values = [features_dict.get(col, 0) for col in feature_columns]
        features_scaled = scaler.transform([feature_values])
        
        prediction = predictor_model.predict(features_scaled)[0]
        probability = predictor_model.predict_proba(features_scaled)[0]
        success_probability = probability[1] * 100
        
        component_scores = calculate_component_scores(model_data)
        component_avg = sum(component_scores.values()) / len(component_scores)
        overall_score = (success_probability * 0.7) + (component_avg * 0.3)
        overall_score = min(100, max(0, overall_score))
        
        if overall_score >= 75:
            category = 'High'
        elif overall_score >= 55:
            category = 'Moderate'
        elif overall_score >= 35:
            category = 'Developing'
        else:
            category = 'Challenging'
        
        insights = generate_insights(model_data, overall_score, prediction)
        
        response = {
            'success': True,
            'overall_score': round(overall_score, 1),
            'prediction': {
                'label': 'Success' if prediction == 1 else 'Needs Improvement',
                'category': category,
                'ml_probability': round(success_probability, 1),
                'raw_prediction': int(prediction)
            },
            'component_scores': {
                'revenue_score': round(component_scores['revenue_score'], 1),
                'growth_score': round(component_scores['growth_score'], 1),
                'engagement_score': round(component_scores['engagement_score'], 1),
                'innovation_score': round(component_scores['innovation_score'], 1),
                'marketing_score': round(component_scores['marketing_score'], 1)
            },
            'insights': insights
        }
        
        print(f"\n✅ Prediction made:")
        print(f"   ML Prediction: {prediction} ({'Success' if prediction == 1 else 'Failure'})")
        print(f"   Probability: {success_probability:.1f}%")
        print(f"   Overall Score: {overall_score:.1f}%")
        
        return jsonify(response)
    
    except Exception as e:
        print(f"❌ Error in prediction: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': predictor_model is not None,
        'features_count': len(feature_columns) if feature_columns else 0
    })


if __name__ == '__main__':
    print("\n" + "="*80)
    print("🚀 AI MICROBUSINESS SUCCESS PREDICTOR API")
    print("="*80 + "\n")
    
    if load_trained_model():
        print("\n✅ Server starting...")
        print("📍 Access the app at: http://localhost:5003")
        print("📍 The web interface will load automatically")
        print("="*80 + "\n")
        
        app.run(debug=True, host='0.0.0.0', port=5003)
    else:
        print("\n❌ Failed to start server. Please train the model first.")
        print("\n💡 To train the model, run:")
        print("   python predictor.py")
        print("\nThis will create:")
        print("   ✓ trained_model.pkl")
        print("   ✓ scaler.pkl")
        print("   ✓ feature_columns.pkl")
        print("="*80 + "\n")