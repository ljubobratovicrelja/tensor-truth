import streamlit as st
import sys
import os
import time
import json  # NEW: Required for persistence

# Add src to path
sys.path.append(os.path.abspath("./src"))
# IMPORT FROM NEW SHARED LIB
from rag_engine import load_inference_index, get_query_engine

# --- CONFIGURATION ---
HISTORY_FILE = "chat_history.json"  # NEW: File path for persistence

st.set_page_config(
    page_title="Tensor-Truth", 
    page_icon="⚡", 
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .stChatMessage { padding: 1rem; border-radius: 10px; }
    .stSpinner { margin-top: 2rem; }
</style>
""", unsafe_allow_html=True)

# --- PERSISTENCE FUNCTIONS (NEW) ---
def load_history():
    """Loads chat history from local JSON file."""
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            # If file is corrupted, return empty list
            return []
    return []

def save_history(messages):
    """Saves current chat history to local JSON file."""
    try:
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(messages, f, indent=2)
    except Exception as e:
        st.error(f"Failed to save history: {e}")

# --- CACHED RESOURCE LOADING ---
@st.cache_resource(show_spinner=False)
def get_global_engine():
    print("--- BOOTING APP ENGINE ---")
    start_time = time.time()
    
    try:
        # This now ONLY loads. It never builds.
        index = load_inference_index() 
        engine = get_query_engine(index)
    except FileNotFoundError as e:
        # Nice error message for the UI
        st.error(f"🚨 Setup Required: {e}")
        st.info("Run `python src/build_db.py` in your terminal first.")
        st.stop()
    
    print(f"--- READY in {time.time() - start_time:.2f}s ---")
    return engine

# --- SIDEBAR CONTROLS ---
with st.sidebar:
    st.header("⚡ Tensor-Truth")
    st.caption(f"Running on: RTX 3090Ti")
    
    with st.status("System Status", expanded=True):
        st.write("✅ **Embeddings:** BGE-M3 (GPU)")
        st.write("✅ **LLM:** DeepSeek-R1:32b")
        st.write("✅ **Vector DB:** Chroma (Persisted)")
        st.write("✅ **History:** Auto-Saving") # NEW
        
    st.divider()
    
    # 2. Session Management
    if st.button("🧹 Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        save_history([]) # NEW: Clear the file on disk too!
        st.rerun()

    st.info("💡 **Note:** History is now saved to disk. It will survive app restarts.")

# --- MAIN APP LOGIC ---

# 1. Load Engine
try:
    with st.spinner("Connecting to Neural Core..."):
        query_engine = get_global_engine()
except Exception as e:
    st.error(f"Critical Error: {e}")
    st.stop()

# 2. Load Chat History (UPDATED)
if "messages" not in st.session_state:
    # Try to load from disk first
    st.session_state.messages = load_history()

# 3. Display Chat
st.title("Local RAG Interface")
if not st.session_state.messages:
    st.markdown("👋 _Ready. Ask about PyTorch, NumPy, or your indexed libraries._")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 4. Input & Processing
if prompt := st.chat_input("How do I..."):
    # Append User Msg
    st.session_state.messages.append({"role": "user", "content": prompt})
    save_history(st.session_state.messages) # NEW: Save immediately
    
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate Response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        try:
            with st.spinner("DeepSeek is thinking..."):
                response = query_engine.query(prompt)
                
                full_response = response.response
                message_placeholder.markdown(full_response)
                
                with st.expander("🔍 Source Context (Reranked)"):
                    for i, node in enumerate(response.source_nodes):
                        score = node.score if node.score else 0.0
                        color = "green" if score > 0.7 else "orange"
                        
                        st.markdown(f"**Node {i+1}** (Score: :{color}[{score:.3f}])")
                        st.caption(f"File: `{node.metadata.get('file_name', 'Unknown')}`")
                        st.code(node.node.get_content()[:400] + "...", language="python")
                        st.divider()

            # Append Assistant Msg
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            save_history(st.session_state.messages) # NEW: Save immediately

        except Exception as e:
            st.error(f"Generation Error: {e}")