import streamlit as st
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    AIMessagePromptTemplate,
    ChatPromptTemplate
)
# Custom CSS styling
st.markdown("""
<style>
            /* Existing styles */
            .main {
                background-color: #1a1a1a;
                color: #ffffff;
            }
            .sidebar .sidebar
            content {
                background-color: #2d2d2d;
            }
            .stTextInput textarea {
                color: #ffffff !important;
            }

            /* Add these new styles for select box */
            .stSelectbox div[data-baseweb="select"] {
                color: white !important;
                background-color: #3d3d3d !important;
            }

            .stSelectbox svg {
                fill: white !important;
            }

            .stSelectbox option {
                background-color: #2d2d2d !important;
                color: white !important;
            }

            /* For dropdown menu items */
            div[role="listbox"] div {
                background-color: #2d2d2d !important;
                color: white !important;
            }
        </style>
        """, unsafe_allow_html=True)
st.title("🧠 DeepSeek Code Companion")
st.caption("🚀 Your AI Pair Programmer with Debugging Superpowers")

# sidebar configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    selected_model = st.selectbox(
        "Choose Model",
        ["deepseek-r1:1.5b"],
        index=0
    )
    st.divider()
    st.markdown("### Model Capabilities") 
    st.markdown("""
    - 🐍 Python Expert
    - 🐞 Debugging Assistant
    - 📝 Code Documentation
    - 💡 Design Solution
    """)
    st.divider()
    st.markdown("Built with [Ollama](https://ollama.ai/) | [LangChain](https://python.langchain.com/)")

# Initiate chat Engine

llm_engine = ChatOllama(
    model = selected_model,
    temperature = 0.3,
    base_url = "http://localhost:11434"
)
# System prompt configuration
system_prompt = SystemMessagePromptTemplate.from_template(
    "You are an expert AI coding assistant. Provide concise, correct solutions "
    "with strategic print statements for debugging. Always respond in English."
)

# Session state management
if "message_log" not in st.session_state:
    st.session_state.message_log = [{"role":"ai","content":"How can I assist you today? 💻"}]

# Container for the chat messages
chat_container = st.container()

# Display chat messages
with chat_container:
    for message in st.session_state.message_log:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# Chat input and processing
user_query = st.chat_input("Type your message here...")

def generate_ai_response(prompt_chain):
    processing_pipeline = prompt_chain | llm_engine | StrOutputParser()
    return processing_pipeline.invoke({})

def build_prompt_chain():
    prompt_sequence = [system_prompt]
    for msg in st.session_state.message_log:
        if msg["role"] == "user":
            prompt_sequence.append(HumanMessagePromptTemplate.from_template(msg["content"]))
        elif msg["role"] == "ai":
            prompt_sequence.append(AIMessagePromptTemplate.from_template(msg["content"]))
    return ChatPromptTemplate.from_messages(prompt_sequence)

if user_query:
    # Adding user message to log
    st.session_state.message_log.append({"role": "user", "content": user_query})

    # Generate AI response
    with st.spinner("🧠 Generating response..."):
        prompt_chain = build_prompt_chain()
        ai_response = generate_ai_response(prompt_chain)

    # Adding AI response to log
    st.session_state.message_log.append({"role": "ai", "content" : ai_response})

    # Rerun to update the chat messages
    st.rerun()