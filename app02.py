# app.py

import streamlit as st
import joblib
from preprocessing_module import preprocess_text

# Load model and vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Set Streamlit page configuration
st.set_page_config(page_title="Nigeria Fake News Detector", layout="centered", page_icon="📢")

# --- Logo and Title ---
st.markdown("""
    <div style='text-align: center;'>
        <img src='https://www.itedgenews.africa/wp-content/uploads/2024/02/3MTT.jpg' width='200'/>
        <h1 style='color: #00753a;'>Nigeria Fake News Detection System</h1>
        <p><i>Combating misinformation, one headline at a time.</i></p>
    </div>
""", unsafe_allow_html=True)

# --- Input Section ---
st.subheader("📰 Analyze a News Headline or Article")

# ✅ Tagline: Protect Yourself from Fake News...
st.markdown("""
    <div style='text-align: center; font-size: 0.9em; color: gray;'>
        <i>Protect Yourself from Fake News, Legal Risks & Let's Build a Peaceful Nation.<br>
        Verify before Amplify.</i>
    </div>
""", unsafe_allow_html=True)

user_input = st.text_area("Paste your news headline or paragraph here:")

if st.button("Analyze Now"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text to analyze.")
    else:
        try:
            processed_text = preprocess_text(user_input)
            vectorized_text = vectorizer.transform([processed_text])
            prediction = model.predict(vectorized_text)[0]  # ✅ Correct index

            if prediction == 0:
                st.error("🔴 **This appears to be a FAKE NEWS**")
            else:
                st.success("🟢 **News Appears Genuine**")

        except Exception as e:
            st.error(f"Prediction failed: {e}")

# --- Legal Education Section ---
st.markdown("---")
st.markdown("""
### ⚖️ Why Every Nigerian Must Use This App

> **The cost of sharing unverified news could be your freedom.**  
> The laws are real. The consequences are personal.

📌 **Cybercrime Act (Section 19 & 24):**  
Sharing fake or deceptive news — even via WhatsApp — is punishable by imprisonment.

📌 **Inciting Violence or Panic:**  
Publishing content that promotes ethnic hatred, fear, or unrest is a criminal offense.

📌 **Stricter Laws Incoming:**  
Proposed legislation like the Fake News Bill will enforce tighter digital content control.

🔐 **How This App Protects You:**

✅ Instantly detect fake or misleading news  
✅ Verify WhatsApp messages before forwarding  
✅ Stay protected, stay informed, stay legal  

🛡️ **Ignorance is no defense.**  
Don’t risk prosecution or jail over something you didn’t verify.  
This tool was built to protect **YOU** — use it **before you share it**.
""")

# --- About Section ---
st.markdown("---")
st.markdown("""
#### 📌 About This Tool
This AI-powered tool uses Natural Language Processing and Machine Learning (TF-IDF + Random Forest) trained on local Nigerian fake and real news datasets to classify headlines or paragraphs as genuine or misleading.

#### 🔍 Why it Matters
Fake news can incite violence, disrupt elections, and polarize society. This tool is a first step in empowering citizens, journalists, and policymakers with technology to promote truth in public discourse.

#### 👨🏽‍💻 Developer
Built by [Tayo Adebimpe (FMCIDE/3MTT Cohort 3. FE/23/65216968)] — passionate about ethical AI, open data, and digital resilience in West Africa.
""")
