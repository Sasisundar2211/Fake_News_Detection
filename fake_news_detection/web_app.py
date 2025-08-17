import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
from fake_news_detector import FakeNewsDetector


detector = FakeNewsDetector()
detector.load_model("fake_news_detector_model.pkl")

st.title("📰 Fake News Detection System")

title = st.text_input("News Title")
text = st.text_area("News Content")
source = st.text_input("News Source (optional)")

if st.button("Analyze"):
    if text.strip() == "":
        st.warning("Please enter the news content.")
    else:
        result = detector.predict(text=text, title=title, source=source)
        st.subheader("Prediction")
        st.write(f"**Result:** {result['prediction']}")
        st.write(f"**Confidence:** {result['confidence']:.2%}")
        st.write("Details:", result['analysis'])
