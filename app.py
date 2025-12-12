import streamlit as st
import sys
import os
import json
import uuid
import gc
import torch
import time  # NEW: For timing
from datetime import datetime

sys.path.append(os.path.abspath("./src"))
from rag_engine import load_engine_for_modules

# --- CONFIG ---
SESSIONS_FILE = "chat_sessions.json"
INDEX_DIR = "./indexes"

st.set_page_config(page_title="Tensor-Truth", layout="wide", page_icon="⚡")

# --- CSS ---
st.markdown("""
<style>
    .stButton button { text-align: left; padding-left: 10px; width: 100%; }
    .stChatMessage { padding: 1rem; border-radius: 10px; }
    div[data-testid="stExpander"] { border: none; box-shadow: none; }
</style>
""", unsafe_allow_html=True)

# --- HELPER: Scan Available Modules ---
def get_available_modules():
    if not os.path.exists(INDEX_DIR): return []
    return sorted([d for d in os.listdir(INDEX_DIR) if os.path.isdir(os.path.join(INDEX_DIR, d))])

# --- MEMORY MANAGEMENT ---
def free_memory():
    if "engine" in st.session_state:
        del st.session_state["engine"]
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("--- VRAM FLUSHED ---")

# CHANGED: Now checks PARAMS as well as MODULES
def ensure_engine_loaded(target_modules, target_params):
    """
    Ensures the loaded engine matches the required modules AND parameters.
    """
    target_tuple = tuple(sorted(target_modules))
    # Create a hashable representation of the params
    param_hash = frozenset(target_params.items())
    
    current_config = st.session_state.get("loaded_config") # (modules, params)

    # If everything matches, return existing engine
    if current_config == (target_tuple, param_hash):
        return st.session_state.engine

    # CONTEXT SWITCH DETECTED
    if current_config is not None:
        placeholder = st.empty()
        placeholder.info("⏳ Re-configuring Neural Engine... (VRAM Flush)")
        free_memory()
        
        try:
            placeholder.info(f"⏳ Loading: {target_modules} | T={target_params['temperature']}")
            engine = load_engine_for_modules(list(target_tuple), target_params)
            
            st.session_state.engine = engine
            st.session_state.loaded_config = (target_tuple, param_hash)
            placeholder.empty()
            return engine
        except Exception as e:
            placeholder.error(f"Failed to load: {e}")
            st.stop()
    else:
        # First load
        try:
            engine = load_engine_for_modules(list(target_tuple), target_params)
            st.session_state.engine = engine
            st.session_state.loaded_config = (target_tuple, param_hash)
            return engine
        except Exception as e:
            st.error(f"Startup Failed: {e}")
            st.stop()

# --- SESSION MANAGEMENT ---
def load_sessions():
    if os.path.exists(SESSIONS_FILE):
        try:
            with open(SESSIONS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except: pass
    return {"current_id": None, "sessions": {}}

def save_sessions():
    with open(SESSIONS_FILE, "w", encoding="utf-8") as f:
        json.dump(st.session_state.chat_data, f, indent=2)

# CHANGED: Now accepts params
def create_session(modules, params):
    new_id = str(uuid.uuid4())
    st.session_state.chat_data["sessions"][new_id] = {
        "title": "New Session",
        "created_at": str(datetime.now()),
        "messages": [],
        "modules": modules,
        "params": params  # Store T and Context Window
    }
    st.session_state.chat_data["current_id"] = new_id
    save_sessions()
    return new_id

def update_title(session_id, text):
    session = st.session_state.chat_data["sessions"][session_id]
    if session.get("title") == "New Session":
        new_title = (text[:30] + '..') if len(text) > 30 else text
        session["title"] = new_title
        save_sessions()

def rename_session(new_title):
    current_id = st.session_state.chat_data.get("current_id")
    if current_id:
        st.session_state.chat_data["sessions"][current_id]["title"] = new_title
        save_sessions()
        st.rerun()

# --- INITIALIZATION ---
if "chat_data" not in st.session_state:
    st.session_state.chat_data = load_sessions()

if "mode" not in st.session_state:
    if st.session_state.chat_data.get("current_id"):
        st.session_state.mode = "chat"
    else:
        st.session_state.mode = "setup"

if "loaded_config" not in st.session_state:
    st.session_state.loaded_config = None # Stores ((modules), (params))
if "engine" not in st.session_state:
    st.session_state.engine = None

# ==========================================
# SIDEBAR
# ==========================================
with st.sidebar:
    st.header("⚡ Tensor-Truth")
    
    if st.button("➕ Start New Chat", type="primary", use_container_width=True):
        st.session_state.mode = "setup"
        st.session_state.chat_data["current_id"] = None
        st.rerun()

    st.divider()
    st.subheader("🗂️ History")
    
    session_ids = list(st.session_state.chat_data["sessions"].keys())
    for sess_id in reversed(session_ids):
        sess = st.session_state.chat_data["sessions"][sess_id]
        title = sess.get("title", "Untitled")
        current_id = st.session_state.chat_data.get("current_id")
        is_active = (sess_id == current_id)
        
        label = f"{'📂' if is_active else '📄'} {title} "
        
        if st.button(label, key=sess_id, use_container_width=True, disabled=is_active):
            st.session_state.chat_data["current_id"] = sess_id
            st.session_state.mode = "chat"
            st.rerun()

    st.divider()
    
    if st.session_state.mode == "chat" and st.session_state.chat_data.get("current_id"):
        curr_id = st.session_state.chat_data["current_id"]
        curr_sess = st.session_state.chat_data["sessions"][curr_id]
        
        with st.expander("⚙️ Session Settings"):
            new_name = st.text_input("Rename:", value=curr_sess.get("title"))
            if st.button("Update"):
                rename_session(new_name)
            
            st.caption("Contexts:")
            for m in curr_sess.get("modules", []):
                st.code(m, language="text")
            
            # Show Params
            p = curr_sess.get("params", {})
            st.caption(f"Temp: {p.get('temperature')} | Ctx: {p.get('context_window')}")

            st.markdown("---")
            if st.button("🗑️ Delete Chat"):
                del st.session_state.chat_data["sessions"][curr_id]
                st.session_state.chat_data["current_id"] = None
                st.session_state.mode = "setup"
                free_memory()
                st.session_state.loaded_config = None
                save_sessions()
                st.rerun()

# ==========================================
# MAIN CONTENT AREA
# ==========================================

# VIEW 1: SETUP SCREEN
if st.session_state.mode == "setup":
    st.title("Initiate New Session")
    
    available = get_available_modules()
    
    with st.form("new_chat_form"):
        st.subheader("1. Knowledge Base")
        selected_mods = st.multiselect("Select Active Indices:", available, default=available)
        
        st.subheader("2. Model Parameters")
        col1, col2 = st.columns(2)
        with col1:
            temp = st.slider("Temperature (Creativity)", 0.0, 1.0, 0.1, 0.1)
        with col2:
            ctx = st.select_slider("Context Window (Tokens)", options=[2048, 4096, 8192, 16384], value=4096)
        
        st.markdown("---")
        submitted = st.form_submit_button("🚀 Launch Chat", type="primary")
        
        if submitted:
            if not selected_mods:
                st.error("Please select at least one index.")
            else:
                # Pack params
                params = {"temperature": temp, "context_window": ctx}
                create_session(selected_mods, params)
                st.session_state.mode = "chat"
                st.rerun()

# VIEW 2: CHAT INTERFACE
elif st.session_state.mode == "chat":
    current_id = st.session_state.chat_data.get("current_id")
    if not current_id:
        st.session_state.mode = "setup"
        st.rerun()
        
    session = st.session_state.chat_data["sessions"][current_id]
    modules = session.get("modules", [])
    params = session.get("params", {"temperature": 0.1, "context_window": 4096}) # Backwards compat
    
    # --- CONTEXT ENFORCER ---
    if modules:
        engine = ensure_engine_loaded(modules, params)
    else:
        st.warning("No linked knowledge base.")
        engine = None

    st.title(session.get("title", "Untitled"))
    
    # --- RENDER HISTORY ---
    for msg in session["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            
            # Render Meta Info (Sources + Time)
            meta_cols = st.columns([3, 1])
            with meta_cols[0]:
                if "sources" in msg and msg["sources"]:
                    with st.expander("📚 Sources"):
                        for src in msg["sources"]:
                            st.caption(f"{src['file']} ({src['score']:.2f})")
            with meta_cols[1]:
                if "time_taken" in msg:
                    st.caption(f"⏱️ {msg['time_taken']:.2f}s")

    # --- CHAT INPUT ---
    if prompt := st.chat_input("Ask..."):
        update_title(current_id, prompt)
        
        with st.chat_message("user"):
            st.markdown(prompt)
        session["messages"].append({"role": "user", "content": prompt})
        save_sessions()
        
        with st.chat_message("assistant"):
            if engine:
                with st.spinner("Thinking..."):
                    # --- TIMER START ---
                    start_time = time.time()
                    
                    response = engine.chat(prompt)
                    
                    # --- TIMER END ---
                    end_time = time.time()
                    elapsed = end_time - start_time
                    
                    st.markdown(response.response)
                    
                    # Capture Sources
                    source_data = []
                    if response.source_nodes:
                        with st.expander("📚 Sources"):
                            for node in response.source_nodes:
                                score = float(node.score) if node.score else 0.0
                                fname = node.metadata.get('file_name', 'Unknown')
                                st.caption(f"{fname} ({score:.2f})")
                                source_data.append({"file": fname, "score": score})
                    
                    # Show Time
                    st.caption(f"⏱️ Generation time: {elapsed:.2f}s")
                    
                    session["messages"].append({
                        "role": "assistant", 
                        "content": response.response,
                        "sources": source_data,
                        "time_taken": elapsed  # <--- SAVE TIME TO HISTORY
                    })
                    save_sessions()
            else:
                st.error("Engine not loaded.")
        
        st.rerun()