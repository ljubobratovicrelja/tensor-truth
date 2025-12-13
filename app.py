import streamlit as st
import sys
import os
import json
import uuid
import gc
import torch
import time
import requests 
from datetime import datetime

sys.path.append(os.path.abspath("./src"))
from rag_engine import load_engine_for_modules
from utils import parse_thinking_response, run_ingestion, convert_chat_to_markdown

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

# --- HELPERS ---
def get_available_modules():
    if not os.path.exists(INDEX_DIR): return []
    return sorted([d for d in os.listdir(INDEX_DIR) if os.path.isdir(os.path.join(INDEX_DIR, d))])

def get_ollama_models():
    """Fetches list of available models from local Ollama instance."""
    try:
        response = requests.get("http://localhost:11434/api/tags")
        if response.status_code == 200:
            models = [m["name"] for m in response.json()["models"]]
            return sorted(models)
    except:
        pass
    return ["deepseek-r1:8b"] 

def free_memory():
    if "engine" in st.session_state:
        del st.session_state["engine"]
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def ensure_engine_loaded(target_modules, target_params):
    target_tuple = tuple(sorted(target_modules))
    param_items = sorted([(k, v) for k, v in target_params.items()])
    param_hash = frozenset(param_items)
    
    current_config = st.session_state.get("loaded_config") 

    if current_config == (target_tuple, param_hash):
        return st.session_state.engine

    if current_config is not None:
        placeholder = st.empty()
        placeholder.info(f"⏳ Loading Model: {target_params.get('model')}... (VRAM Flush)")
        free_memory()
        try:
            engine = load_engine_for_modules(list(target_tuple), target_params)
            st.session_state.engine = engine
            st.session_state.loaded_config = (target_tuple, param_hash)
            placeholder.empty()
            return engine
        except Exception as e:
            placeholder.error(f"Failed: {e}")
            st.stop()
    else:
        try:
            engine = load_engine_for_modules(list(target_tuple), target_params)
            st.session_state.engine = engine
            st.session_state.loaded_config = (target_tuple, param_hash)
            return engine
        except Exception as e:
            st.error(f"Startup Failed: {e}")
            st.stop()

# --- SESSION MGMT ---
def load_sessions():
    if os.path.exists(SESSIONS_FILE):
        try:
            with open(SESSIONS_FILE, "r", encoding="utf-8") as f: return json.load(f)
        except: pass
    return {"current_id": None, "sessions": {}}

def save_sessions():
    with open(SESSIONS_FILE, "w", encoding="utf-8") as f:
        json.dump(st.session_state.chat_data, f, indent=2)

def create_session(modules, params):
    new_id = str(uuid.uuid4())
    st.session_state.chat_data["sessions"][new_id] = {
        "title": "New Session",
        "created_at": str(datetime.now()),
        "messages": [],
        "modules": modules,
        "params": params
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
if "chat_data" not in st.session_state: st.session_state.chat_data = load_sessions()

# FIX: Always default to "setup" on a fresh app load
if "mode" not in st.session_state:
    st.session_state.mode = "setup"

if "loaded_config" not in st.session_state: st.session_state.loaded_config = None
if "engine" not in st.session_state: st.session_state.engine = None

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
            if st.button("Update"): rename_session(new_name)
            
            st.caption("Contexts:")
            for m in curr_sess.get("modules", []): st.code(m, language="text")
            
            p = curr_sess.get("params", {})
            st.caption(f"Model: {p.get('model', 'Unknown')}")
            st.caption(f"Reranker: {p.get('reranker_model', 'Default')} (Top {p.get('reranker_top_n')})")
            st.caption(f"Temp: {p.get('temperature')} | Ctx: {p.get('context_window')} | Conf: {p.get('confidence_cutoff')}")
            
            md_data = convert_chat_to_markdown(curr_sess)
            st.download_button("📥 Export to Markdown", md_data, f"{curr_sess['title'][:20]}.md", "text/markdown")

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

if st.session_state.mode == "setup":
    st.title("Control Center")
    
    tab_launch, tab_ingest = st.tabs(["🚀 Launch Session", "📥 Library Ingestion"])
    
    with tab_launch:
        available_mods = get_available_modules()
        available_models = get_ollama_models()
        
        # Determine defaults
        default_model_idx = 0
        for i, m in enumerate(available_models):
            if "deepseek-r1:8b" in m: default_model_idx = i
        
        with st.form("new_chat_form"):
            st.markdown("### 🚀 Start a New Research Session")
            st.caption("Default Configuration: All Available Indices • DeepSeek-R1 • Temp 0.3")
            
            # COLLAPSED SETTINGS (The 'Dropdown' Effect)
            with st.expander("⚙️ Configure Settings (Indices, Model, Params)", expanded=False):
                
                col_a, col_b = st.columns(2)
                with col_a:
                    st.subheader("Knowledge Base")
                    selected_mods = st.multiselect("Active Indices:", available_mods, default=[])
                
                with col_b:
                    st.subheader("Model Selection")
                    selected_model = st.selectbox("LLM:", available_models, index=default_model_idx)
                    sys_prompt = st.text_area("System Instructions:", height=68, placeholder="Optional...")

                st.subheader("3. Retrieval Strategy (Reranker)")
                r1, r2, r3 = st.columns(3)
                with r1:
                    reranker_model = st.selectbox(
                        "Reranker Model", 
                        options=["BAAI/bge-reranker-v2-m3", "BAAI/bge-reranker-base", "cross-encoder/ms-marco-MiniLM-L-6-v2"],
                        index=0
                    )
                with r2:
                    top_n = st.number_input("Top N (Final Context)", min_value=1, max_value=20, value=3)
                with r3:
                    conf = st.slider("Confidence Cutoff", 0.0, 1.0, 0.3, 0.05)

                st.subheader("Parameters")
                c1, c2 = st.columns(2)
                with c1:
                    temp = st.slider("Temperature", 0.0, 1.0, 0.3, 0.1)
                with c2:
                    ctx = st.select_slider("Context Window", options=[2048, 4096, 8192, 16384], value=4096)
            
            st.markdown("---")
            
            # MAIN ACTION BUTTON
            if st.form_submit_button("Start Session", type="primary", use_container_width=True):
                if not selected_mods:
                    st.error("Please select at least one index.")
                else:
                    params = {
                        "model": selected_model,
                        "temperature": temp, 
                        "context_window": ctx,
                        "system_prompt": sys_prompt,
                        "reranker_model": reranker_model,
                        "reranker_top_n": top_n,
                        "confidence_cutoff": conf
                    }
                    create_session(selected_mods, params)
                    st.session_state.mode = "chat"
                    st.rerun()

    with tab_ingest:
        st.subheader("Add Papers to Library")
        arxiv_input = st.text_input("ArXiv ID (e.g., 2310.06825):")
        if st.button("Fetch & Index"):
            if not arxiv_input:
                st.warning("Please enter an ID.")
            else:
                progress = st.empty()
                progress.info("⏳ Starting ingestion pipeline...")
                success, logs = run_ingestion(arxiv_input)
                for log in logs: st.text(log)
                if success:
                    st.success("Ingestion Complete!")
                    time.sleep(2)
                    st.rerun()

elif st.session_state.mode == "chat":
    current_id = st.session_state.chat_data.get("current_id")
    if not current_id:
        st.session_state.mode = "setup"
        st.rerun()
        
    session = st.session_state.chat_data["sessions"][current_id]
    modules = session.get("modules", [])
    params = session.get("params", {"model": "deepseek-r1:8b", "temperature": 0.3, "context_window": 4096, "confidence_cutoff": 0.3})
    
    if modules:
        engine = ensure_engine_loaded(modules, params)
    else:
        st.warning("No linked knowledge base.")
        engine = None

    st.title(session.get("title", "Untitled"))
    
    for msg in session["messages"]:
        with st.chat_message(msg["role"]):
            thought, answer = parse_thinking_response(msg["content"])
            if thought:
                with st.expander("💭 Thought Process", expanded=False): st.markdown(thought)
            st.markdown(answer)
            
            meta_cols = st.columns([3, 1])
            with meta_cols[0]:
                if "sources" in msg and msg["sources"]:
                    with st.expander("📚 Sources"):
                        for src in msg["sources"]: st.caption(f"{src['file']} ({src['score']:.2f})")
            with meta_cols[1]:
                if "time_taken" in msg: st.caption(f"⏱️ {msg['time_taken']:.2f}s")

    if prompt := st.chat_input("Ask..."):
        update_title(current_id, prompt)
        
        with st.chat_message("user"): st.markdown(prompt)
        session["messages"].append({"role": "user", "content": prompt})
        save_sessions()
        
        with st.chat_message("assistant"):
            if engine:
                with st.spinner(f"Thinking ({params.get('model')})..."):
                    start_time = time.time()
                    response = engine.chat(prompt)

                    if not response.source_nodes and response.response.strip() == "Empty Response":
                        raw_content = (
                            "I analyzed the knowledge base, but I could not find any documents "
                            "with sufficient relevance to answer this query.\n\n"
                            "I am strictly constrained to answer **only** from the provided context."
                        )

                        # Clear thought since we didn't run the model
                        thought = None 
                        answer = raw_content
                    else:
                        # Standard processing
                        raw_content = response.response
                        thought, answer = parse_thinking_response(raw_content)

                    elapsed = time.time() - start_time
                    
                    if thought:
                        with st.expander("💭 Thought Process", expanded=True): st.markdown(thought)
                    st.markdown(answer)
                    
                    source_data = []
                    if response.source_nodes:
                        with st.expander("📚 Sources"):
                            for node in response.source_nodes:
                                score = float(node.score) if node.score else 0.0
                                fname = node.metadata.get('file_name', 'Unknown')
                                st.caption(f"{fname} ({score:.2f})")
                                source_data.append({"file": fname, "score": score})
                    
                    st.caption(f"⏱️ {elapsed:.2f}s")
                    session["messages"].append({
                        "role": "assistant", 
                        "content": raw_content, 
                        "sources": source_data,
                        "time_taken": elapsed
                    })
                    save_sessions()
            else:
                st.error("Engine not loaded.")
        st.rerun()