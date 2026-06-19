import streamlit as st
import joblib

# Page Config
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>

.stApp {
    background: linear-gradient(135deg, #141E30, #243B55);
}

.main-title {
    text-align: center;
    color: white;
    font-size: 50px;
    font-weight: bold;
    margin-bottom: 10px;
}

.sub-title {
    text-align: center;
    color: #d3d3d3;
    font-size: 18px;
    margin-bottom: 30px;
}

.card {
    background-color: rgba(255,255,255,0.08);
    padding: 30px;
    border-radius: 20px;
    backdrop-filter: blur(10px);
    box-shadow: 0px 4px 20px rgba(0,0,0,0.3);
}

.stTextArea textarea {
    border-radius: 12px;
    font-size: 16px;
}

.stButton > button {
    width: 100%;
    background: linear-gradient(90deg,#00C9FF,#92FE9D);
    color: black;
    border-radius: 12px;
    height: 50px;
    font-size: 18px;
    font-weight: bold;
    border: none;
}

.stButton > button:hover {
    transform: scale(1.03);
    transition: 0.3s;
}

</style>
""", unsafe_allow_html=True)

# Load Model
try:
    vectorizer = joblib.load("vectorizer.jb")
    model = joblib.load("lr_model.jb")

except Exception as e:
    st.error(f"Error Loading Model: {e}")
    st.stop()

# Header
st.markdown(
    '<div class="main-title">📰 Fake News Detector</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-title">AI Powered News Verification System</div>',
    unsafe_allow_html=True
)

# Center Layout
col1, col2, col3 = st.columns([1,2,1])

with col2:

    st.markdown('<div class="card">', unsafe_allow_html=True)

    news_text = st.text_area(
        "Paste News Article Here",
        height=250,
        placeholder="Enter news article text..."
    )

    if st.button("🔍 Analyze News"):

        if news_text.strip():

            transformed = vectorizer.transform([news_text])
            prediction = model.predict(transformed)

            st.markdown("---")

            if prediction[0] == 1:
                st.success("✅ This News Appears To Be REAL")
                st.balloons()

            else:
                st.error("🚨 This News Appears To Be FAKE")

        else:
            st.warning("⚠️ Please Enter News Content First")

    st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown(
    """
    <br><br>
    <center>
    <p style='color:lightgray'>
    Built with ❤️ using Streamlit & Machine Learning
    </p>
    </center>
    """,
    unsafe_allow_html=True
)