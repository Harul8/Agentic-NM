import streamlit as st
from crew import crew

st.title("⚖️ Legal AI System")

facts = st.text_area("Enter your case facts:")

if st.button("Analyze Case"):
    result = crew.kickoff(inputs={"raw_facts": facts})
    st.subheader("Legal Analysis Output")
    st.write(result)
