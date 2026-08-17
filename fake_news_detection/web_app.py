"""Optional Streamlit interface for a locally trained demonstration model."""

from pathlib import Path

import streamlit as st

from fake_news_detector import NewsAnalyzer


MODEL_PATH = Path(__file__).with_name("fake_news_detector_model.pkl")


@st.cache_resource
def load_detector() -> NewsAnalyzer | None:
    """Load the optional local model without pretending it ships with the app."""
    if not MODEL_PATH.exists():
        return None

    detector = NewsAnalyzer()
    detector.load_trained_model(MODEL_PATH)
    return detector


st.title("Fake News Detection — Educational Demo")
st.caption("Uses a locally trained classical-ML model. It is not a fact-checking service.")

detector = load_detector()
if detector is None:
    st.info("No trained local model was found. Run the demonstration before using this interface.")
    st.stop()

title = st.text_input("News title")
text = st.text_area("News content")
source = st.text_input("News source (optional)")

if st.button("Analyze"):
    if not text.strip():
        st.warning("Enter news content before running the demonstration.")
    else:
        result = detector.analyze_article(text, headline=title, source_name=source)
        st.subheader("Demonstration result")
        st.write(f"**Prediction:** {result['prediction']}")
        st.write(f"**Confidence:** {result['confidence']:.2%}")
        st.json(result["content_analysis"])
