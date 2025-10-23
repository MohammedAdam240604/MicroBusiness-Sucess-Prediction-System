from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import pickle
import os
import requests
import json

app = Flask(__name__)
CORS(app)

# Global variables
predictor_model = None
scaler = None
feature_columns = None

# Ollama configuration
OLLAMA_BASE_URL = 'http://localhost:11434'
OLLAMA_MODEL = 'llama3'  # Change to your preferred model (e.g., 'mistral', 'neural-chat', etc.)

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


def check_ollama_available():
    """Check if Ollama is running and available"""
    try:
        response = requests.get(f'{OLLAMA_BASE_URL}/api/tags', timeout=2)
        return response.status_code == 200
    except:
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


def query_ollama(prompt, context=""):
    """Query Ollama for AI-generated response"""
    try:
        # Add instruction for concise responses
        full_prompt = f"{context}\n\nUser: {prompt}\n\nAssistant: Please provide a concise, focused response (2-3 paragraphs max) addressing the user's question directly."
        
        response = requests.post(
            f'{OLLAMA_BASE_URL}/api/generate',
            json={
                'model': OLLAMA_MODEL,
                'prompt': full_prompt,
                'stream': False,
                'temperature': 0.7,
                'top_p': 0.9,
                'top_k': 40
            },
            timeout=120  # Increased timeout for longer processing
        )
        
        if response.status_code == 200:
            result = response.json()
            return {
                'success': True,
                'response': result.get('response', 'No response generated'),
                'model': OLLAMA_MODEL
            }
        else:
            print(f"Ollama error: {response.status_code}")
            print(f"Response: {response.text}")
            return {
                'success': False,
                'error': f'Ollama returned status code {response.status_code}: {response.text}'
            }
    
    except requests.exceptions.Timeout:
        return {
            'success': False,
            'error': 'Ollama request timed out. Try a smaller model like "mistral": ollama pull mistral'
        }
    except Exception as e:
        print(f"Query error: {e}")
        return {
            'success': False,
            'error': str(e)
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
            'business_data': model_data,  # Store for chat context
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
            'insights': insights,
            'ollama_available': check_ollama_available()
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


@app.route('/api/chat', methods=['POST'])
def chat():
    """Chat endpoint for AI-powered business improvement suggestions"""
    try:
        data = request.json
        user_message = data.get('message', '')
        business_data = data.get('business_data', {})
        prediction_results = data.get('prediction_results', {})
        
        if not user_message:
            return jsonify({
                'success': False,
                'error': 'Message cannot be empty'
            }), 400
        
        if not check_ollama_available():
            return jsonify({
                'success': False,
                'error': 'Ollama is not running. Please start Ollama first: ollama serve'
            }), 503
        
        # Build context from business data and predictions
        context = f"""You are an expert business consultant analyzing a micro-business in India.

BUSINESS ANALYSIS CONTEXT:
- Business Type: {business_data.get('type', 'Unknown')}
- Monthly Revenue: ₹{business_data.get('monthlyRevenue', 0):,}
- Customer Count: {business_data.get('customerCount', 0)}
- Growth Rate: {business_data.get('growthRate', 0)}%
- Team Size: {business_data.get('teamSize', 0)}
- Digital Marketing Reach: {business_data.get('marketingReach', 0)}
- Customer Engagement Score: {business_data.get('engagementScore', 0)}/10
- Innovation Score: {business_data.get('innovationScore', 0)}/10
- Sustainability Index: {business_data.get('sustainabilityIndex', 0)}/10
- Founder Experience: {business_data.get('founderExperience', 0)} years
- Operating Period: {business_data.get('operatingMonths', 0)} months
- Competition Level: {business_data.get('competitionLevel', 'Medium')}

PREDICTION RESULTS:
- Overall Success Score: {prediction_results.get('overall_score', 0)}%
- Category: {prediction_results.get('category', 'Unknown')}

IDENTIFIED STRENGTHS:
{chr(10).join([f"• {s}" for s in prediction_results.get('strengths', [])])}

IDENTIFIED WEAKNESSES:
{chr(10).join([f"• {w}" for w in prediction_results.get('weaknesses', [])])}

INITIAL RECOMMENDATIONS:
{chr(10).join([f"• {r}" for r in prediction_results.get('recommendations', [])])}

Please provide specific, actionable advice based on the user's question and this business context."""
        
        # Query Ollama
        result = query_ollama(user_message, context)
        
        if result['success']:
            return jsonify({
                'success': True,
                'response': result['response'],
                'model': result['model']
            })
        else:
            return jsonify({
                'success': False,
                'error': result['error']
            }), 500
    
    except Exception as e:
        print(f"❌ Error in chat: {e}")
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
        'features_count': len(feature_columns) if feature_columns else 0,
        'ollama_available': check_ollama_available()
    })
@app.route('/api/visualizations', methods=['POST'])
def get_visualizations():
    """Generate visualization data for charts"""
    try:
        data = request.json
        model_data = map_html_to_model_format(data)
        
        # Score breakdown for radar/bar chart
        component_scores = calculate_component_scores(model_data)
        score_breakdown = [
            {'category': 'Revenue', 'score': round(component_scores['revenue_score'], 1)},
            {'category': 'Growth', 'score': round(component_scores['growth_score'], 1)},
            {'category': 'Engagement', 'score': round(component_scores['engagement_score'], 1)},
            {'category': 'Innovation', 'score': round(component_scores['innovation_score'], 1)},
            {'category': 'Marketing', 'score': round(component_scores['marketing_score'], 1)}
        ]
        
        # Revenue projection timeline (based on growth rate)
        months = min(model_data.get('Operating_Months', 12), 12)
        current_revenue = model_data.get('Monthly_Revenue', 0)
        growth_rate = model_data.get('Growth_Rate', 0) / 100
        
        revenue_timeline = []
        for i in range(months + 1):
            # Calculate historical revenue (reverse projection)
            historical_revenue = current_revenue / ((1 + growth_rate) ** (months - i))
            revenue_timeline.append({
                'month': f'M{i}',
                'revenue': round(historical_revenue, 0)
            })
        
        # Comparison with industry benchmarks
        comparison_data = [
            {
                'metric': 'Revenue',
                'current': model_data.get('Monthly_Revenue', 0),
                'benchmark': 30000
            },
            {
                'metric': 'Customers',
                'current': model_data.get('Customer_Count', 0),
                'benchmark': 200
            },
            {
                'metric': 'Growth %',
                'current': model_data.get('Growth_Rate', 0),
                'benchmark': 15
            },
            {
                'metric': 'Team Size',
                'current': model_data.get('Team_Size', 0),
                'benchmark': 8
            },
            {
                'metric': 'Marketing',
                'current': model_data.get('Digital_Marketing_Reach', 0),
                'benchmark': 10000
            }
        ]
        
        # Feature importance (mock data - you can use actual model feature_importances_)
        if predictor_model and hasattr(predictor_model, 'feature_importances_'):
            feature_importance = []
            importances = predictor_model.feature_importances_
            for i, col in enumerate(feature_columns[:8]):  # Top 8 features
                feature_importance.append({
                    'feature': col.replace('_', ' ').title(),
                    'importance': round(float(importances[i]) * 100, 1)
                })
            feature_importance.sort(key=lambda x: x['importance'], reverse=True)
        else:
            feature_importance = [
                {'feature': 'Revenue', 'importance': 25},
                {'feature': 'Growth Rate', 'importance': 20},
                {'feature': 'Engagement', 'importance': 18},
                {'feature': 'Innovation', 'importance': 15},
                {'feature': 'Marketing', 'importance': 12},
                {'feature': 'Team Size', 'importance': 10}
            ]
        
        return jsonify({
            'success': True,
            'score_breakdown': score_breakdown,
            'revenue_timeline': revenue_timeline,
            'comparison_data': comparison_data,
            'feature_importance': feature_importance
        })
        
    except Exception as e:
        print(f"❌ Error generating visualizations: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    print("\n" + "="*80)
    print("🚀 AI MICROBUSINESS SUCCESS PREDICTOR API (with Ollama Integration)")
    print("="*80 + "\n")
    
    if load_trained_model():
        print("\n✅ Server starting...")
        print("📍 Access the app at: http://localhost:5003")
        print("="*80 + "\n")
        
        # Check Ollama
        if check_ollama_available():
            print("✅ Ollama is running and available!")
            print(f"   Model: {OLLAMA_MODEL}")
        else:
            print("⚠️  Ollama is not running")
            print("   Chat features will be unavailable")
            print("   To enable, run: ollama serve")
        
        print("\n" + "="*80 + "\n")
        
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