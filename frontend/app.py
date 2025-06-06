import streamlit as st
import requests

API_URL = "http://localhost:8000"

st.title("Job Matchmaking Agent")

option = st.selectbox("Choose an action", ["Match Resumes", "Match Resumes (LangGraph)", "Chat with AI"])

if option == "Match Resumes":
    jd = st.text_area("Enter Job Description")
    if st.button("Find Matches"):
        if jd.strip():
            response = requests.post(f"{API_URL}/match-resumes", json={"text": jd})
            if response.status_code == 200:
                matches = response.json().get("matches", [])
                st.write("Top matching resumes:")
                for res, dist in matches:
                    st.write(f"- {res} (Distance: {dist:.4f})")
            else:
                st.error("Error from backend")
        else:
            st.warning("Please enter a job description")

elif option == "Match Resumes (LangGraph)":
    jd = st.text_area("Enter Job Description")
    if st.button("Find Matches with LangGraph"):
        if jd.strip():
            response = requests.post(f"{API_URL}/graph-match", json={"text": jd})
            if response.status_code == 200:
                data = response.json()
                st.write("Top Matches:")
                for match in data["matches"]:
                    st.write(f"- {match[0]} (Score: {match[1]:.4f})")
                st.markdown("### LLM Summary")
                st.write(data["llm_summary"])
            else:
                st.error("Error from backend")
        else:
            st.warning("Please enter a job description")

elif option == "Chat with AI":
    prompt = st.text_input("Enter your prompt")
    if st.button("Send"):
        if prompt.strip():
            response = requests.post(f"{API_URL}/chat", json={"prompt": prompt})
            if response.status_code == 200:
                st.write("AI Response:")
                st.write(response.json().get("response", ""))
            else:
                st.error("Error from backend")
        else:
            st.warning("Please enter a prompt")
