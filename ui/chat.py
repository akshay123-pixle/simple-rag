import streamlit as st
import requests

BACKEND_URL = "http://127.0.0.1:8001/query"

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Enterprise Assistant AJ",
    page_icon="🤖",
    layout="centered",
)

st.title("🤖 Enterprise Assistant AJ")

# ---------------- SESSION ----------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------------- DISPLAY CHAT ----------------
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ---------------- CHAT INPUT ----------------
if prompt := st.chat_input("Type your message..."):

    # Show user message
    st.session_state.messages.append(
        {"role": "user", "content": prompt}
    )

    with st.chat_message("user"):
        st.markdown(prompt)

    # Fetch response from FastAPI backend
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                res = requests.post(
                    BACKEND_URL,
                    json={"q": prompt},
                    timeout=30
                )
                if res.status_code == 200:
                    data = res.json()
                    response = data.get("answer", "No answer returned.")
                else:
                    response = f"⚠️ Backend Error ({res.status_code}): {res.text}"
            except Exception:
                response = f"⚠️ Could not connect to backend server at `{BACKEND_URL}`. Please make sure the FastAPI server is running."

        st.markdown(response)

    st.session_state.messages.append(
        {"role": "assistant", "content": response}
    )