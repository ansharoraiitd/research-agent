#Section 1: imports and config 
import streamlit as st 
import requests
import json 

#my live GCP API URL: 
API_URL = "https://research-agent-513976967636.us-central1.run.app"

#Page config: MUST be the first streamlit command  
#Sets the browser tab title and page layout 
st.set_page_config(
    page_title="Research Agent",
    page_icon="🔬",
    layout="centered"
)

#Section 2: page header 
#Header section: 
st.title("🔬 Research Agent")
st.markdown("**Powered by a 4-agent AI pipeline** — Planner → Researcher → Writer → Critic")
st.divider()

#Section 3: Input section- where user types their research question 
st.subheader("What would you like to research?")

task = st.text_area(
    label="Research topic",
    placeholder="e.g. How is LangGraph being used in production AI in 2026?",
    height=100,
    max_chars=300,
    help="Enter a research topic or a question. Max 300 characters."
)

#Character counter - shows how many characters the user has typed
if task:
    st.caption(f"{len(task)}/300 characters")

#Section 4: the button and API call 
#Button row: 
col1, col2 = st.columns([3, 1])

with col1:
    search_clicked = st.button(
        "🔍 Research",
        type="primary",
        use_container_width=True,
        disabled=not task.strip() if task else True 
    )

with col2:
    st.link_button("API docs", f"{API_URL}/docs", use_container_width=True)

st.divider()

#This block runs when the user clicks Research 
if search_clicked and task and task.strip():

    #Show a spinner while waiting for the API
    with st.spinner("Agent is working... (this takes 10-15 seconds)"):
        try:
            #Call my live GCP API
            response = requests.post(
                f"{API_URL}/research",
                json={"task": task.strip()},
                timeout=120
            )

            if response.status_code == 200:
                data = response.json()
                st.session_state["last_result"] = data 
                st.session_state["last_task"] = task.strip()

            else: 
                st.error(f"API error {response.status_code}: {response.json().get('detail', 'Unknown error')}")

        except requests.exceptions.Timeout:
            st.error("Request timed out. The agent took too long. Please try again.")
        except requests.exceptions.ConnectionError:
            st.error("Could not connect to the API. Check your internet connection.")

#Section 5: display the results 

if "last_result" in st.session_state:
    result = st.session_state["last_result"]
    task_shown = st.session_state.get("last_task", "")

    #Metrics row - show stats about the research 
    st.subheader("📊 Research Complete") 
    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Duration", f"{result.get('duration_seconds', 0)}s")
    with m2:
        word_count = len(result.get('report', '').split())
        st.metric("Report Length", f"{word_count} words")
    with m3:
        st.metric("Status", "✅ Complete")

    st.divider()

    #The actual report - rendered as markdown
    st.subheader("📄 Research Report")
    st.markdown(result.get("report", "No report generated"))

    st.divider()

    #Download  button - lets user save the report 
    st.download_button(
        label="⬇️ Download Report",
        data=result.get("report", ""),
        file_name=f"research_{task_shown[:30].replace(' ', '_')}.md",
        mime="text/markdown"
    )

#Section 6: sidebar with info
#Sidebar- always visibleon the left 
with st.sidebar:
    st.header("About")
    st.markdown("""
This tool uses a **4-agent AI pipeline**:

1. 🗺️ **Planner** — creates research questions
2. 🔍 **Researcher** — finds information
3. ✍️ **Writer** — writes structured report
4. ✅ **Critic** — reviews and approves quality

Built with LangChain, LangGraph, FastAPI, and Streamlit.
Deployed on GCP Cloud Run.
    """)    

    st.divider()
    st.caption(f"API: {API_URL}")