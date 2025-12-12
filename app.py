import streamlit as st
import sys
import os
import json
import uuid
from datetime import datetime

sys.path.append(os.path.abspath("./src"))
from rag_engine import load_engine_for_modules

# --- CONFIG ---
SESSIONS_FILE = "chat_sessions.json"
INDEX_DIR = "./indexes"

st.set_page_config(page_title="Tensor-Truth Modular", layout="wide", page_icon="⚡")

# --- CSS FOR UI POLISH ---
st.markdown("""
<style>
    .stButton button {
        text-align: left;
        padding-left: 10px;
        width: 100%;
    }
    .stChatMessage { padding: 1rem; border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

# --- HELPER: Scan Available Modules ---
def get_available_modules():
    if not os.path.exists(INDEX_DIR):
        return []
    return [d for d in os.listdir(INDEX_DIR) if os.path.isdir(os.path.join(INDEX_DIR, d))]

# --- SESSION MANAGEMENT ---
def load_sessions():
    if os.path.exists(SESSIONS_FILE):
        try:
            with open(SESSIONS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except: pass
    return {"current_id": str(uuid.uuid4()), "sessions": {}}

def save_sessions():
    with open(SESSIONS_FILE, "w", encoding="utf-8") as f:
        json.dump(st.session_state.chat_data, f, indent=2)

def create_new_session():
    new_id = str(uuid.uuid4())
    st.session_state.chat_data["sessions"][new_id] = {
        "title": "New Chat",
        "created_at": str(datetime.now()),
        "messages": []
    }
    st.session_state.chat_data["current_id"] = new_id
    save_sessions()

def update_title_if_new(session_id, user_message):
    session = st.session_state.chat_data["sessions"][session_id]
    if session.get("title") == "New Chat":
        new_title = (user_message[:30] + '..') if len(user_message) > 30 else user_message
        session["title"] = new_title
        save_sessions()

def rename_session(new_title):
    current_id = st.session_state.chat_data["current_id"]
    if current_id in st.session_state.chat_data["sessions"]:
        st.session_state.chat_data["sessions"][current_id]["title"] = new_title
        save_sessions()
        st.rerun()

# --- INITIALIZATION ---
if "chat_data" not in st.session_state:
    st.session_state.chat_data = load_sessions()

# Ensure valid current_id
if not st.session_state.chat_data.get("current_id"):
    create_new_session()

# State for Engine
if "engine" not in st.session_state:
    st.session_state.engine = None
if "active_modules" not in st.session_state:
    st.session_state.active_modules = []

# ==========================================
# SIDEBAR: CONFIGURATION
# ==========================================
with st.sidebar:
    st.header("⚡ Tensor-Truth")
    
    # 1. Module Selector
    with st.expander("📚 Knowledge Base", expanded=True):
        available = get_available_modules()
        if not available:
            st.error("No indices found! Run `build_db.py` first.")
        
        selected = st.multiselect(
            "Select Active Contexts:", 
            available, 
            default=available 
        )
        
        if st.button("🚀 Load / Reload Engine", type="primary"):
            with st.spinner("Mounting Vector Databases..."):
                try:
                    st.session_state.engine = load_engine_for_modules(selected)
                    st.session_state.active_modules = selected
                    st.success("Engine Online!")
                except Exception as e:
                    st.error(f"Failed: {e}")

    st.divider()
    
    # 2. Current Chat Management (Rename / Delete)
    # We get the current session title safely
    current_id = st.session_state.chat_data.get("current_id")
    if current_id and current_id in st.session_state.chat_data["sessions"]:
        current_title = st.session_state.chat_data["sessions"][current_id].get("title", "Untitled")
        
        with st.expander("⚙️ Active Chat Settings", expanded=False):
            # Rename Input
            new_title = st.text_input("Rename Chat:", value=current_title)
            if st.button("Update Title"):
                rename_session(new_title)
            
            # Delete Button
            st.markdown("---")
            if st.button("🗑️ Delete This Chat", type="secondary"):
                del st.session_state.chat_data["sessions"][current_id]
                remaining = list(st.session_state.chat_data["sessions"].keys())
                st.session_state.chat_data["current_id"] = remaining[-1] if remaining else None
                save_sessions()
                st.rerun()

    st.divider()

    # 3. History List
    st.subheader("🗂️ History")
    
    if st.button("➕ New Chat", use_container_width=True):
        create_new_session()
        st.rerun()

    # List Sessions
    session_ids = list(st.session_state.chat_data["sessions"].keys())
    current_id = st.session_state.chat_data["current_id"]

    for sess_id in reversed(session_ids):
        sess = st.session_state.chat_data["sessions"][sess_id]
        title = sess.get("title", "Untitled")
        
        if sess_id == current_id:
            st.button(f"📂 {title}", key=sess_id, disabled=True, use_container_width=True)
        else:
            if st.button(f"📄 {title}", key=sess_id, use_container_width=True):
                st.session_state.chat_data["current_id"] = sess_id
                st.rerun()
    
    # Nuke Button at the very bottom
    if st.button("🔥 Nuke All History"):
        st.session_state.chat_data = {"current_id": None, "sessions": {}}
        if os.path.exists(SESSIONS_FILE): os.remove(SESSIONS_FILE)
        st.rerun()

# ==========================================
# MAIN PAGE
# ==========================================

# 1. Check if Engine is Loaded
if st.session_state.engine is None:
    st.title("⚡ Tensor-Truth")
    st.info("👈 Please select your modules in the sidebar and click 'Load Engine' to start.")
    st.stop()

# 2. Get Current Session
current_id = st.session_state.chat_data.get("current_id")
if not current_id or current_id not in st.session_state.chat_data["sessions"]:
    create_new_session()
    st.rerun()

session = st.session_state.chat_data["sessions"][current_id]

# 3. Render Header
st.title(f"{session.get('title', 'New Chat')}")
st.caption(f"Active Contexts: {', '.join(st.session_state.active_modules)}")

# 4. Render History
for msg in session["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 5. Handle Input
if prompt := st.chat_input("Ask..."):
    update_title_if_new(current_id, prompt)

    with st.chat_message("user"):
        st.markdown(prompt)
    
    session["messages"].append({"role": "user", "content": prompt})
    save_sessions()
    
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = st.session_state.engine.chat(prompt)
            st.markdown(response.response)
            
            if response.source_nodes:
                with st.expander("Src"):
                    for node in response.source_nodes:
                        st.caption(f"{node.metadata.get('file_name')} ({node.score:.2f})")
            
            session["messages"].append({"role": "assistant", "content": response.response})
            save_sessions()
            
    st.rerun()