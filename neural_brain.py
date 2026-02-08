# -*- coding: utf-8 -*-
# ==============================================================================
#   PROJECT: CM-X NEURAL LINK (GEMINI BRAIN CONNECTION)
#   MODULE: DIGITAL BRAIN
#   AUTHOR: Boss Manikandan & Chellakili
#   PURPOSE: Streamlit-Optimized AI Market Analysis
# ==============================================================================

import streamlit as st
import google.generativeai as genai
import json

class ChellakiliBrain:
    def __init__(self):
        """
        மூளையை தயார் செய்தல் (Initializing the Brain)
        """
        self.api_key = None
        self.model = None
        
        # 1. Streamlit Secrets-ல் இருந்து கீயை எடுத்தல்
        try:
            if "GEMINI_API_KEY" in st.secrets:
                self.api_key = st.secrets["GEMINI_API_KEY"]
                genai.configure(api_key=self.api_key)
                # வேகமான முடிவுகளுக்கு Flash மாடல் சிறந்தது
                self.model = genai.GenerativeModel('gemini-1.5-flash')
            else:
                st.error("❌ Neural Link Broken: GEMINI_API_KEY not found in secrets!")
        except Exception as e:
            st.warning(f"⚠️ Brain Initialization Error: {e}")

    def analyze_market(self, price, rsi, trend, fiis_data, physics_velocity):
        """
        மார்க்கெட் நிலவரத்தை அலசி, JSON வடிவில் பதில் தரும்.
        """
        # கீ இல்லையென்றால் உடனே நிறுத்து
        if not self.model:
            return {
                "decision": "WAIT",
                "confidence": "LOW",
                "reason": "Brain offline (Check API Key)"
            }

        # 2. மூளைக்கான கட்டளை (Prompt Engineering)
        prompt = f"""
        Act as 'Chellakili', an elite scalping AI for Indian Nifty 50.
        
        LIVE DATA:
        - Price: {price}
        - RSI: {rsi}
        - Trend: {trend}
        - Velocity ($v$): {physics_velocity} (Physics Engine)
        - FII Data: {fiis_data}
        
        LOGIC:
        - Velocity dropping + Price rising = TRAP (Sell/Avoid).
        - Velocity rising + Price rising = STRONG BUY.
        - RSI > 75 is Overbought (Be careful).
        
        OUTPUT (Strict JSON):
        {{
            "decision": "BUY_CE" or "BUY_PE" or "WAIT",
            "confidence": "HIGH" or "LOW",
            "reason": "Short explanation in Tanglish (Tamil+English)"
        }}
        """

        try:
            # 3. ஏஐ-யிடம் கேள்வி கேட்டல்
            response = self.model.generate_content(prompt)
            
            # 4. பதிலை சுத்தம் செய்தல் (Clean Response)
            clean_text = response.text.replace('```json', '').replace('```', '').strip()
            
            # JSON-ஆக மாற்றுதல்
            decision_data = json.loads(clean_text)
            return decision_data
            
        except Exception as e:
            # ஏதாவது பிழை ஏற்பட்டால் பாதுகாப்பான பதில்
            return {
                "decision": "WAIT",
                "confidence": "LOW",
                "reason": f"System Glitch: {str(e)}"
            }

# --- TESTING (ஸ்ட்ரீம்லிட்டில் இதை தனியா ரன் பண்ணி பார்க்க) ---
if __name__ == "__main__":
    st.title("🧠 Neural Link Test")
    brain = ChellakiliBrain()
    
    if st.button("Test Brain Logic"):
        # டம்மி டேட்டா
        result = brain.analyze_market(19500, 78, "UP", "Selling", -2.5)
        st.json(result)
      
