# 🌟 MicroBusiness Success Prediction System with Ollama Llama 3 Chat Integration

## 🧠 Overview
**MicroBusiness Success Prediction System** is a Machine Learning + AI-powered web platform that predicts the **success probability of microbusinesses** and provides **AI-driven business insights** through an integrated **Llama 3 chatbot** (via Ollama).  

The system uses **Flask** for the backend, **Ollama** for Llama 3-based chat responses, and **Machine Learning models** for predictive analytics.

---

## 🎯 Objectives
- Predict whether a microbusiness will succeed based on business attributes.  
- Provide instant business advice or insights through Llama 3 chatbot.  
- Offer visualization and interpretation of key business indicators.  
- Support real-time interaction and ML-driven recommendations.

---

## 🚀 Features
✅ Predict business success probability using ML models  
✅ Real-time AI chat powered by **Ollama + Llama 3**  
✅ Flask backend with REST API endpoints (`/api/predict`, `/api/chat`, `/api/visualizations`)  
✅ Clean React / HTML frontend (interactive and responsive)  
✅ Data visualization dashboard for insights  
✅ Model interpretability (feature importance, charts)  

---

## 🧩 Tech Stack
| Category | Tools / Frameworks |
|-----------|--------------------|
| **Backend** | Flask (Python) |
| **AI Chat** | Ollama + Llama 3 |
| **ML / Data Science** | Scikit-learn, Pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn, Plotly |
| **Frontend** | React / HTML / CSS / JS |
| **Version Control** | Git & GitHub |

---

## ⚙️ Setup Instructions

### Step 1️⃣ Clone the Repository
```bash
git clone https://github.com/MohammedAdam240604/MicroBusiness-Sucess-Prediction-System.git
cd MicroBusiness-Sucess-Prediction-System
Step 2️⃣ Create and Activate Virtual Environment
bash
Copy code
python -m venv venv
# Activate
venv\Scripts\activate       # Windows
source venv/bin/activate    # Mac/Linux
Step 3️⃣ Install Dependencies
bash
Copy code
pip install -r requirements.txt
Step 4️⃣ Install and Run Ollama
Download Ollama: https://ollama.com/download

Open a terminal and start the Ollama server:

bash
Copy code
ollama serve
Pull the Llama 3 model:

bash
Copy code
ollama pull llama3
Verify:

bash
Copy code
ollama list
Step 5️⃣ Run the Flask Backend
bash
Copy code
python app.py
Runs at: http://127.0.0.1:5000

🧠 Chat with Llama 3
Once the backend and Ollama are running, you can send a POST request to the chat API:

Endpoint:
bash
Copy code
POST /api/chat
Example Request:
json
Copy code
{
  "message": "What are the key factors for a microbusiness to succeed in India?"
}
Example Response:
json
Copy code
{
  "reply": "The key factors include founder experience, funding adequacy, customer reach, and regional economic activity."
}
🧮 Predict Business Success
Send a POST request to:

bash
Copy code
POST /api/predict
Example Payload:
json
Copy code
{
  "location": "Coimbatore",
  "industry": "Retail",
  "experience_years": 3,
  "funding": 40000,
  "employees": 4
}
Example Response:
json
Copy code
{
  "success_probability": 0.81,
  "prediction": "High Success Potential",
  "important_features": ["Experience", "Funding", "Industry"]
}
📊 Visualization Endpoint
bash
Copy code
POST /api/visualizations
Generates dynamic charts for:

Feature importance

Prediction distribution

Model accuracy metrics

📂 Folder Structure
php
Copy code
MicroBusiness-Sucess-Prediction-System/
│
├── app.py                 # Flask backend with ML + Ollama integration
├── models/                # Saved ML model files
├── frontend/              # React / HTML UI
├── data/                  # Input datasets
├── static/, templates/    # Flask assets
├── requirements.txt       # Dependencies
└── README.md              # Documentation
🧠 Architecture Diagram
Frontend (React/HTML) ⟶ Flask API (app.py) ⟶
➡ ML Model (Predict Success)
➡ Ollama + Llama 3 (AI Chat Responses)
➡ Visualization Module (Charts & Insights)

🛠 Future Enhancements
Deploy Flask + Ollama backend on cloud (Render/AWS)

Add conversation history and context memory for chatbot

Integrate user login and personalized recommendations

Improve model accuracy with advanced ensemble techniques

🧑‍💻 Author
Mohammed Adam H
B.Tech — Computer Science and Engineering
Hindustan Institute of Technology and Science

🪪 License
MIT License — free to use, modify, and distribute with attribution.

💬 Acknowledgements
Dataset derived from OpenStreetMap and public microbusiness data.

Llama 3 model served locally via Ollama.

Inspired by real-world business intelligence systems.

⭐ If you find this project useful, please give it a star on GitHub!

yaml
Copy code

---

Would you like me to include **badges** (for Python version, Flask, Ollama, license, and repo stars) at the top — like a professional open-source project header?






