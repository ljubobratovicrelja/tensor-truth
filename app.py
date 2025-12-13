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
from utils import parse_thinking_response, run_ingestion, convert_chat_to_markdown, get_running_models

# --- CONFIG ---
SESSIONS_FILE = "chat_sessions.json"
PRESETS_FILE = "presets.json"
INDEX_DIR = "./indexes"
MAX_VRAM_GB = 24.0  # RTX 3090 Ti Limit (Configurable)

st.set_page_config(page_title="Tensor-Truth", layout="wide", page_icon="⚡")

# --- CSS ---
st.markdown("""
<style>
    .stButton button { text-align: left; padding-left: 10px; width: 100%; }
    .stChatMessage { padding: 1rem; border-radius: 10px; }
    div[data-testid="stExpander"] { border: none; box-shadow: none; }
    code { color: #d63384; }
</style>
""", unsafe_allow_html=True)

# --- HELPERS ---

@st.cache_data(ttl=10)
def get_available_modules():
    if not os.path.exists(INDEX_DIR): return []
    return sorted([d for d in os.listdir(INDEX_DIR) if os.path.isdir(os.path.join(INDEX_DIR, d))])

@st.cache_data(ttl=60)
def get_ollama_models():
    """Fetches list of available models from local Ollama instance."""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=1)
        if response.status_code == 200:
            models = [m["name"] for m in response.json()["models"]]
            return sorted(models)
    except:
        pass
    return ["deepseek-r1:8b"] 

def get_system_devices():
    """Returns list of available compute devices."""
    devices = ["cpu"]
    # Check MPS (Apple Silicon)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        devices.insert(0, "mps")
    # Check CUDA
    if torch.cuda.is_available():
        devices.insert(0, "cuda")
    return devices

def free_memory():
    if "engine" in st.session_state:
        del st.session_state["engine"]
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        torch.mps.empty_cache()

# --- PRESET MANAGEMENT ---
def load_presets():
    if os.path.exists(PRESETS_FILE):
        try:
            with open(PRESETS_FILE, "r", encoding="utf-8") as f: return json.load(f)
        except: pass
    return {}

def save_preset(name, config):
    presets = load_presets()
    presets[name] = config
    with open(PRESETS_FILE, "w", encoding="utf-8") as f:
        json.dump(presets, f, indent=2)

def delete_preset(name):
    presets = load_presets()
    if name in presets:
        del presets[name]
        with open(PRESETS_FILE, "w", encoding="utf-8") as f:
            json.dump(presets, f, indent=2)

def apply_preset(name, available_mods, available_models, available_devices):
    presets = load_presets()
    if name not in presets: return

    p = presets[name]
    
    # Update Session State Keys directly
    
    # 1. Modules
    valid_mods = [m for m in p.get("modules", []) if m in available_mods]
    st.session_state.setup_mods = valid_mods
    
    # 2. Model
    model = p.get("model")
    if model in available_models:
        st.session_state.setup_model = model
        
    # 3. Parameters
    st.session_state.setup_reranker = p.get("reranker_model", "BAAI/bge-reranker-v2-m3")
    st.session_state.setup_ctx = p.get("context_window", 4096)
    st.session_state.setup_temp = p.get("temperature", 0.3)
    st.session_state.setup_top_n = p.get("reranker_top_n", 3)
    st.session_state.setup_conf = p.get("confidence_cutoff", 0.3)
    st.session_state.setup_sys_prompt = p.get("system_prompt", "")
    
    # 4. Devices
    rag_dev = p.get("rag_device", "cpu")
    if rag_dev in available_devices:
        st.session_state.setup_rag_device = rag_dev
        
    llm_dev = p.get("llm_device", "gpu")
    if llm_dev in ["cpu", "gpu"]:
        st.session_state.setup_llm_device = llm_dev

@st.cache_data(ttl=2, show_spinner=False)
def get_vram_breakdown():
    """
    Returns detailed VRAM stats:
    - total_used: What Task Manager says
    - reclaimable: What we can kill (Ollama + PyTorch)
    - baseline: What stays (OS, Browser, Display)
    """
    if not torch.cuda.is_available():
        # Heuristic for non-CUDA systems (like Mac)
        return {"total_used": 0.0, "reclaimable": 0.0, "baseline": 4.0}
    
    try:
        # 1. Real Hardware Usage (Everything on the card)
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        total_used_gb = (total_bytes - free_bytes) / (1024**3)
        
        # 2. PyTorch Reserved (What THIS python process holds)
        torch_reserved_gb = torch.cuda.memory_reserved() / (1024**3)
        
        # 3. Ollama Usage (External process)
        ollama_usage_gb = 0.0
        active_models = get_running_models() 
        for m in active_models:
            try:
                size_str = m.get('size_vram', '0 GB').split()[0]
                ollama_usage_gb += float(size_str)
            except: pass
            
        reclaimable = torch_reserved_gb + ollama_usage_gb
        baseline = max(0.5, total_used_gb - reclaimable)
        
        return {
            "total_used": total_used_gb,
            "reclaimable": reclaimable,
            "baseline": baseline
        }
    except Exception:
        return {"total_used": 0.0, "reclaimable": 0.0, "baseline": 2.5}

def estimate_vram_usage(model_name, num_indices, context_window, rag_device, llm_device):
    """
    Returns (predicted_total, breakdown_dict, new_session_cost)
    """
    stats = get_vram_breakdown()
    system_baseline = stats["baseline"]
    
    # --- RAG COST ---
    # 1.8GB if running on GPU (CUDA) or MPS (Unified Memory counts as VRAM usage)
    if rag_device in ["cuda", "mps"]:
        rag_overhead = 1.8 
    else:
        rag_overhead = 0.0 # CPU RAM
    
    index_overhead = num_indices * 0.15 
    
    # --- LLM COST ---
    if llm_device == "cpu":
        llm_size = 0.0
    else:
        name = model_name.lower() if model_name else ""
        if "70b" in name: llm_size = 40.0
        elif "32b" in name: llm_size = 19.0
        elif "14b" in name: llm_size = 9.5
        elif "8b" in name: llm_size = 5.5
        elif "7b" in name: llm_size = 5.0
        elif "1.5b" in name: llm_size = 1.5
        else: llm_size = 6.0 
    
    # KV Cache (Linear Approx)
    kv_cache = (context_window / 4096) * 0.8
    # KV Cache also moves to RAM if LLM is on CPU
    if llm_device == "cpu": kv_cache = 0.0

    new_session_cost = rag_overhead + index_overhead + llm_size + kv_cache
    predicted_total = system_baseline + new_session_cost
    
    return predicted_total, stats, new_session_cost

def render_vram_gauge(model_name, num_indices, context_window, rag_device, llm_device):
    predicted, stats, new_cost = estimate_vram_usage(model_name, num_indices, context_window, rag_device, llm_device)
    vram_percent = min(predicted / MAX_VRAM_GB, 1.0)
    
    current_used = stats["total_used"]
    reclaimable = stats["reclaimable"]
    
    # Visual Layout
    st.markdown("##### 🖥️ VRAM Status")
    
    m1, m2, m3 = st.columns(3)
    m1.metric("Current Load", f"{current_used:.1f} GB", delta_color="off")
    m2.metric("Reclaimable", f"{reclaimable:.1f} GB", help="VRAM from Ollama/Torch that will be freed.", delta_color="normal")
    m3.metric("Predicted Peak", f"{predicted:.1f} GB", delta=f"{predicted - MAX_VRAM_GB:.1f} GB" if predicted > MAX_VRAM_GB else "Safe", delta_color="inverse")

    color = "green"
    if vram_percent > 0.75: color = "orange"
    if vram_percent > 0.95: color = "red"
    
    st.progress(vram_percent)
    
    if predicted > MAX_VRAM_GB: 
        st.error(f"🛑 Configuration ({predicted:.1f} GB) exceeds limit ({MAX_VRAM_GB} GB).")
    elif predicted > (MAX_VRAM_GB * 0.9): 
        st.warning("⚠️ High VRAM usage predicted.")
    
    return predicted

def ensure_engine_loaded(target_modules, target_params):
    target_tuple = tuple(sorted(target_modules))
    param_items = sorted([(k, v) for k, v in target_params.items()])
    param_hash = frozenset(param_items)
    
    current_config = st.session_state.get("loaded_config") 

    if current_config == (target_tuple, param_hash):
        return st.session_state.engine

    if current_config is not None:
        placeholder = st.empty()
        placeholder.info(f"⏳ Loading Model: {target_params.get('model')} | Pipeline: {target_params.get('rag_device')} | LLM: {target_params.get('llm_device')}...")
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

def generate_smart_title(text, model_name):
    """
    Uses the local LLM to generate a concise title.
    Returns the generated title or a truncated fallback.
    """
    try:
        # Prompt designed to minimize fluff
        prompt = f"Summarize this query into a concise 3-5 word title. Return ONLY the title text, no quotes. Query: {text}"
        
        payload = {
            "model": model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_ctx": 1024, # Low context for speed
                "num_predict": 20 # Short answer
            }
        }
        
        # Direct API call to avoid spinning up full engine logic
        resp = requests.post("http://localhost:11434/api/generate", json=payload, timeout=5)
        if resp.status_code == 200:
            raw = resp.json().get("response", "")
            
            # Clean reasoning traces if model uses them (e.g. DeepSeek-R1)
            _, clean = parse_thinking_response(raw)
            
            # Final cleanup
            title = clean.replace('"', '').replace("'", "").replace(".", "").strip()
            if title:
                return title
    except Exception as e:
        # Silently fail to fallback
        pass
    
    # Fallback
    return (text[:30] + '..') if len(text) > 30 else text

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

def update_title(session_id, text, model_name):
    session = st.session_state.chat_data["sessions"][session_id]
    if session.get("title") == "New Session":
        new_title = generate_smart_title(text, model_name)
        session["title"] = new_title
        save_sessions()

def rename_session(new_title):
    current_id = st.session_state.chat_data.get("current_id")
    if current_id:
        st.session_state.chat_data["sessions"][current_id]["title"] = new_title
        save_sessions()
        st.rerun()

def process_command(prompt, session):
    """Handles /slash commands."""
    cmd_parts = prompt.strip().split()
    command = cmd_parts[0].lower()
    args = cmd_parts[1:] if len(cmd_parts) > 1 else []
    
    active_mods = session.get("modules", [])
    available_mods = get_available_modules()
    current_params = session.get("params", {})
    available_devices = get_system_devices()

    response_msg = ""
    
    if command in ["/list", "/ls", "/status"]:
        lines = ["### 📚 Knowledge Base & System Status"]
        for mod in available_mods:
            lines.append(f"- {'✅' if mod in active_mods else '⚪'} {mod}")
        
        lines.append(f"\n**Pipeline Device:** `{current_params.get('rag_device', 'cuda')}`")
        lines.append(f"**LLM Device:** `{current_params.get('llm_device', 'gpu')}`")
        lines.append("\n**Usage:** `/load <name>`, `/device rag <cpu|cuda|mps>`, `/device llm <cpu|gpu>`")
        response_msg = "\n".join(lines)

    elif command == "/help":
        lines = [
            "### 🛠️ Command Reference",
            "- **/list** / **/status**: Show active indices & hardware usage.",
            "- **/load <index>**: Load a specific knowledge base.",
            "- **/unload <index>**: Unload a knowledge base.",
            "- **/reload**: Flush VRAM and restart the engine.",
            "- **/device rag <cpu|cuda|mps>**: Move RAG pipeline (Embed/Rerank) to specific hardware.",
            "- **/device llm <cpu|gpu>**: Move LLM (Ollama) to specific hardware.",
            "- **/help**: Show this list."
        ]
        response_msg = "\n".join(lines)

    elif command == "/load":
        if not args:
            response_msg = "⚠️ Usage: `/load <index_name>`"
        else:
            target = args[0]
            if target not in available_mods:
                response_msg = f"❌ Index `{target}` not found."
            elif target in active_mods:
                response_msg = f"ℹ️ Index `{target}` is active."
            else:
                session["modules"].append(target)
                save_sessions()
                st.session_state.loaded_config = None 
                response_msg = f"✅ **Loaded:** `{target}`. Engine restarting..."
                st.rerun()

    elif command == "/unload":
        if not args:
            response_msg = "⚠️ Usage: `/unload <index_name>`"
        else:
            target = args[0]
            if target not in active_mods:
                response_msg = f"ℹ️ Index `{target}` not active."
            else:
                session["modules"].remove(target)
                save_sessions()
                st.session_state.loaded_config = None
                response_msg = f"✅ **Unloaded:** `{target}`. Engine restarting..."
                st.rerun()

    elif command == "/reload":
        free_memory()
        st.session_state.loaded_config = None
        response_msg = "🔄 **System Reload:** Memory flushed."
        st.rerun()

    elif command == "/device":
        if len(args) < 2:
            response_msg = "⚠️ Usage: `/device rag <cpu|cuda|mps>` OR `/device llm <cpu|gpu>`"
        else:
            target_type = args[0].lower() # 'rag' or 'llm'
            target_dev = args[1].lower() # 'cpu', 'cuda', ...

            if target_type == "rag":
                if target_dev not in available_devices:
                    response_msg = f"❌ Device `{target_dev}` not available. Options: {available_devices}"
                else:
                    session["params"]["rag_device"] = target_dev
                    save_sessions()
                    st.session_state.loaded_config = None
                    response_msg = f"⚙️ **Pipeline Switched:** Now running Embed/Rerank on `{target_dev.upper()}`."
                    st.rerun()
            
            elif target_type == "llm":
                if target_dev not in ["cpu", "gpu"]:
                    response_msg = "❌ LLM Device options: `cpu` or `gpu`"
                else:
                    session["params"]["llm_device"] = target_dev
                    save_sessions()
                    st.session_state.loaded_config = None
                    response_msg = f"⚙️ **LLM Switched:** Now running Model on `{target_dev.upper()}`."
                    st.rerun()
            else:
                response_msg = "❌ Unknown target. Use `rag` or `llm`."

    else:
        return False, None

    return True, response_msg

# --- INITIALIZATION ---
if "chat_data" not in st.session_state: st.session_state.chat_data = load_sessions()
if "mode" not in st.session_state: st.session_state.mode = "setup"
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
        
        with st.expander("⚙️ Session Settings", expanded=True):
            p = curr_sess.get("params", {})
            mods = curr_sess.get("modules", [])
            
            # Dynamic Gauge (Outside form in chat mode, ok)
            render_vram_gauge(
                p.get('model', 'Unknown'), 
                len(mods), 
                p.get('context_window', 4096),
                p.get('rag_device', 'cuda'),
                p.get('llm_device', 'gpu')
            )
            
            st.divider()
            
            new_name = st.text_input("Rename:", value=curr_sess.get("title"))
            if st.button("Update Title"): rename_session(new_name)
            
            st.caption("Active Indices:")
            if not mods: st.caption("*None*")
            for m in mods: st.code(m, language="text")
            
            md_data = convert_chat_to_markdown(curr_sess)
            st.download_button("📥 Export", md_data, f"{curr_sess['title'][:20]}.md", "text/markdown")

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
        # 1. Fetch Data
        available_mods = get_available_modules()
        available_models = get_ollama_models()
        system_devices = get_system_devices()
        presets = load_presets()
        
        default_model_idx = 0
        for i, m in enumerate(available_models):
            if "deepseek-r1:8b" in m: default_model_idx = i
        
        # 2. Initialize Widget State if New
        if "setup_init" not in st.session_state:
            try:
                cpu_index = system_devices.index("cpu")
            except ValueError:
                cpu_index = 0
            
            st.session_state.setup_mods = []
            st.session_state.setup_model = available_models[default_model_idx] if available_models else None
            st.session_state.setup_reranker = "BAAI/bge-reranker-v2-m3"
            st.session_state.setup_ctx = 4096
            st.session_state.setup_temp = 0.3
            st.session_state.setup_top_n = 3
            st.session_state.setup_conf = 0.3
            st.session_state.setup_sys_prompt = ""
            # Default RAG to CPU
            st.session_state.setup_rag_device = "cpu" if "cpu" in system_devices else "cuda"
            st.session_state.setup_llm_device = "gpu"
            st.session_state.setup_init = True
            
        st.markdown("### 🚀 Start a New Research Session")
        st.caption("Configure your knowledge base and model parameters.")

        # --- PRESETS SECTION ---
        if presets:
            with st.expander("📁 Saved Configurations (Presets)", expanded=True):
                col_p1, col_p2, col_p3 = st.columns([3, 1, 1])
                with col_p1:
                    selected_preset = st.selectbox("Select Preset:", list(presets.keys()), label_visibility="collapsed")
                with col_p2:
                    if st.button("📂 Load", use_container_width=True):
                        apply_preset(selected_preset, available_mods, available_models, system_devices)
                        st.rerun()
                with col_p3:
                    if st.button("🗑️ Delete", type="primary", use_container_width=True):
                        delete_preset(selected_preset)
                        st.rerun()

        # --- FORM WRAPPER FOR STABILITY ---
        with st.form("launch_form"):
            # --- SELECTION AREA ---
            col_a, col_b = st.columns(2)
            with col_a:
                st.subheader("1. Knowledge Base")
                selected_mods = st.multiselect("Active Indices:", available_mods, key="setup_mods")
            
            with col_b:
                st.subheader("2. Model Selection")
                if available_models:
                    selected_model = st.selectbox("LLM:", available_models, key="setup_model")
                else:
                    st.error("No models found in Ollama.")
                    selected_model = "None"
                
            st.subheader("3. RAG Parameters")
            p1, p2, p3 = st.columns(3)
            with p1:
                reranker_model = st.selectbox(
                    "Reranker", 
                    options=["BAAI/bge-reranker-v2-m3", "BAAI/bge-reranker-base", "cross-encoder/ms-marco-MiniLM-L-6-v2"],
                    key="setup_reranker"
                )
            with p2:
                ctx = st.select_slider("Context Window", options=[2048, 4096, 8192, 16384, 32768], key="setup_ctx")
            with p3:
                temp = st.slider("Temperature", 0.0, 1.0, step=0.1, key="setup_temp")

            # Persist expander state
            if "expander_open" not in st.session_state: st.session_state.expander_open = False
            
            # Using custom callback to toggle expander state logic not possible easily with st.expander,
            # but wrapping in form essentially freezes it anyway until submit.
            # We just leave it as standard expander, the form prevents the 'close on slider' behavior.
            with st.expander("Advanced Settings"):
                top_n = st.number_input("Top N (Final Context)", min_value=1, max_value=20, key="setup_top_n")
                conf = st.slider("Confidence Cutoff", 0.0, 1.0, step=0.05, key="setup_conf")
                sys_prompt = st.text_area("System Instructions:", height=68, placeholder="Optional...", key="setup_sys_prompt")
                
                st.markdown("#### Hardware Allocation")
                h1, h2 = st.columns(2)
                
                with h1:
                    rag_device = st.selectbox(
                        "Pipeline Device (Embed/Rerank)", 
                        options=system_devices, 
                        help="Run Retrieval on specific hardware. CPU saves VRAM but is slower.",
                        key="setup_rag_device"
                    )
                with h2:
                    llm_device = st.selectbox(
                        "Model Device (Ollama)",
                        options=["gpu", "cpu"],
                        help="Force Ollama to run on CPU to save VRAM for other tasks.",
                        key="setup_llm_device"
                    )

            st.markdown("---")
            
            # VRAM GAUGE (Inside Form = Updates on Submit)
            st.caption("Click 'Refresh Estimate' to update resource calculation based on current form selections.")
            
            # Use session state values for the gauge to ensure it persists visual state
            # Note: inside form, we see current session state, but new widget values aren't committed to it until submit.
            # So the gauge will always lag one step behind unless we rely on submit.
            vram_est = render_vram_gauge(
                st.session_state.setup_model, 
                len(st.session_state.setup_mods), 
                st.session_state.setup_ctx, 
                st.session_state.setup_rag_device, 
                st.session_state.setup_llm_device
            )
            
            c_btn1, c_btn2 = st.columns([1, 1])
            with c_btn1:
                submitted_check = st.form_submit_button("🔄 Refresh Estimate", use_container_width=True)
            with c_btn2:
                submitted_start = st.form_submit_button("🚀 Start Session", type="primary", use_container_width=True)

            if submitted_check:
                # Just doing this triggers a rerun, updating session_state with new values, 
                # which then redraws the gauge above with correct data.
                pass

            if submitted_start:
                if not selected_mods:
                    st.error("Please select at least one index.")
                elif vram_est > (MAX_VRAM_GB + 4.0): 
                    st.error(f"Config is extremely heavy ({vram_est:.1f}GB). Reduce parameters.")
                else:
                    params = {
                        "model": selected_model,
                        "temperature": temp, 
                        "context_window": ctx,
                        "system_prompt": sys_prompt,
                        "reranker_model": reranker_model,
                        "reranker_top_n": top_n,
                        "confidence_cutoff": conf,
                        "rag_device": rag_device,
                        "llm_device": llm_device
                    }
                    create_session(selected_mods, params)
                    st.session_state.mode = "chat"
                    st.rerun()

        # --- SAVE PRESET SECTION (Outside form to allow name typing without submit) ---
        with st.expander("💾 Save Configuration as Preset"):
            col_s1, col_s2 = st.columns([3, 1])
            with col_s1:
                new_preset_name = st.text_input("Preset Name", placeholder="e.g. 'Deep Search 32B'")
            with col_s2:
                st.write("") # Spacer
                st.write("") 
                if st.button("Save", use_container_width=True):
                    if new_preset_name:
                        # Grab values from session state since they are bound to widgets
                        config_to_save = {
                            "modules": st.session_state.setup_mods,
                            "model": st.session_state.setup_model,
                            "reranker_model": st.session_state.setup_reranker,
                            "context_window": st.session_state.setup_ctx,
                            "temperature": st.session_state.setup_temp,
                            "reranker_top_n": st.session_state.setup_top_n,
                            "confidence_cutoff": st.session_state.setup_conf,
                            "system_prompt": st.session_state.setup_sys_prompt,
                            "rag_device": st.session_state.setup_rag_device,
                            "llm_device": st.session_state.setup_llm_device
                        }
                        save_preset(new_preset_name, config_to_save)
                        st.success(f"Saved: {new_preset_name}")
                        time.sleep(1)
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
                success, logs = run_ingestion("papers", arxiv_input)
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
    params = session.get("params", {"model": "deepseek-r1:8b", "temperature": 0.3, "context_window": 4096, "confidence_cutoff": 0.2})
    
    # Engine loading with visual feedback
    if modules:
        engine = ensure_engine_loaded(modules, params)
    else:
        st.warning("No linked knowledge base. Use `/load <name>` to attach one.")
        engine = None

    st.title(session.get("title", "Untitled"))
    st.caption("💡 Tip: Type **/help** to see all commands. Use `/device` to manage hardware.")
    
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

    if prompt := st.chat_input("Ask or type /cmd..."):
        
        # 1. COMMAND PROCESSING
        if prompt.startswith("/"):
            is_cmd, response = process_command(prompt, session)
            if is_cmd:
                session["messages"].append({"role": "assistant", "content": response})
                save_sessions()
                st.rerun()
        
        # 2. STANDARD CHAT PROCESSING
        # Note: update_title uses the active model for summarization
        update_title(current_id, prompt, params.get("model"))
        
        with st.chat_message("user"): st.markdown(prompt)
        session["messages"].append({"role": "user", "content": prompt})
        save_sessions()
        
        with st.chat_message("assistant"):
            if engine:
                with st.spinner(f"Thinking ({params.get('model')})..."):
                    start_time = time.time()
                    try:
                        response = engine.chat(prompt)

                        if not response.source_nodes and response.response.strip() == "Empty Response":
                            raw_content = "I could not find relevant context in the loaded indices."
                            thought = None 
                            answer = raw_content
                        else:
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
                    except Exception as e:
                        st.error(f"Engine Error: {e}")
            else:
                st.error("Engine not loaded. Use `/load <index>` to start.")
        st.rerun()