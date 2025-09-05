import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import pandas as pd
import time
import re
import json

st.set_page_config(page_title="MODEL PERFORMANCE EVALUATOR", page_icon="📊", layout="wide")
st.title("📊 Model Performance Evaluator")
st.subheader("Find the optimal model through parameter tuning")

# ==== Evaluation prompts definition ====
EVALUATION_PROMPTS = {
    "한국어 능력": [
        "한국의 전통 음식 3가지를 설명해주세요.",
        "존댓말과 반말의 차이점을 예시와 함께 설명하세요."
    ],
    "논리적 추론": [
        "모든 새는 날 수 있다. 펭귄은 새다. 펭귄은 날 수 있는가? 논리적으로 설명하세요.",
        "1, 1, 2, 3, 5, 8, ? 다음 숫자는 무엇인가요?"
    ],
    "창의성": [
        "외계인이 지구를 방문했을 때의 이야기를 3문장으로 써주세요.",
        "시간여행이 가능하다면 어디로 가고 싶나요? 이유도 함께 설명하세요."
    ],
    "정보 추출": [
        "다음 텍스트에서 이메일 주소를 JSON 배열로 추출하세요: 연락처는 john@example.com이고 support@company.co.kr입니다.",
        "다음 문장에서 날짜를 찾아주세요: 우리는 2024년 12월 25일에 만날 예정입니다."
    ],
    "수학/계산": [
        "25 × 37을 계산해주세요.",
        "다음 방정식을 풀어주세요: 2x + 5 = 13"
    ]
}

# ==== Sidebar settings ====
with st.sidebar:
    st.header("🔧 Model Settings")
    MODEL_NAME = st.selectbox(
        "MODEL",
        [
            "yanolja/EEVE-Korean-2.8B-v1.0",
            "nlpai-lab/KULLM3",
            "EleutherAI/polyglot-ko-5.8b",
        ],
        index=0
    )

    precision = st.selectbox(
        "Precision",
        ["4bit", "8bit", "fp16", "fp32"],
        index=0,
        help="Use 4bit/8bit if memory is limited, fp16/fp32 if sufficient"
    )

    st.header("🎛️ Generation Parameters")
    st.info("**Note**: These parameters directly affect model performance!")

    temperature = st.slider("Temperature", 0.1, 2.0, 0.7, 0.1,
                           help="Creativity control (low=consistent, high=creative)")
    top_p = st.slider("Top-p", 0.1, 1.0, 0.9, 0.1,
                     help="Diversity control (low=conservative, high=diverse)")
    max_new_tokens = st.slider("Max Tokens", 50, 500, 256, 10,
                              help="Maximum response length")
    repetition_penalty = st.slider("Repetition Penalty", 1.0, 2.0, 1.10, 0.05,
                                  help="Reduce repetition (>1 recommended)")

    st.write("### Current Parameter Settings")
    st.code(
        f"Model: {MODEL_NAME}\n"
        f"Precision: {precision}\n"
        f"Temperature: {temperature}\n"
        f"Top-p: {top_p}\n"
        f"Max Tokens: {max_new_tokens}\n"
        f"Repetition Penalty: {repetition_penalty}"
    )

    st.markdown("---")
    st.header("🧪 Performance Evaluation")

    # ✅ Baseline / Sweep switch
    baseline_mode = st.checkbox(
        "Baseline mode (single model, deterministic decoding)", value=True,
        help="do_sample=False, T=0.0, top_p=1.0, rep=1.1, Max Tokens fixed"
    )

    # Baseline Max Tokens (128/256)
    baseline_max_tokens = st.selectbox(
        "Baseline Max Tokens", [128, 256], index=0,
        help="128 recommended (short QA). Use 256 if needed."
    )

    eval_mode = st.checkbox("Enable evaluation mode", value=False)

    # Sweep mode toggle
    param_test_mode = st.checkbox("Parameter sweep test (RAG presets)", value=False,
                                  help="do_sample=True with RAG recommended presets (temp/topP/repetition/tokens=128·256)")

    # (Optional) Custom prompts input
    use_benchmark = st.checkbox("Use standard benchmark", value=True,
                               help="If checked, use EVALUATION_PROMPTS below")
    if not use_benchmark:
        eval_prompts_text = st.text_area(
            "Custom prompts (one per line)",
            height=120,
            placeholder="Enter one per line.\nEx)\nExplain in Korean.\nSolve a math problem.",
            disabled=not eval_mode
        )

    run_eval = st.button("🚀 Start Performance Test", disabled=not eval_mode)

    if st.button("🗑️ Clear Cache"):
        st.cache_resource.clear()
        st.success("Cache cleared")

# ==== Model loading ====
@st.cache_resource(show_spinner=True)
def load_model(name: str, mode: str):
    # Tokenizer
    tok = AutoTokenizer.from_pretrained(name, use_fast=True)

    # Common options (adjust numbers for your environment)
    has_cuda = torch.cuda.is_available()
    device_map = "auto" if has_cuda else "cpu"
    max_memory = None
    if has_cuda:
        # Set GPU and CPU memory limits (adjust if needed)
        max_memory = {0: "7GiB", "cpu": "32GiB"}

    offload_kwargs = dict(
        offload_folder="offload",      # Disk offloading folder
        offload_state_dict=True,       # Also offload state dict
    )

    if mode == "4bit":
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16 if has_cuda else torch.float32,
            bnb_4bit_use_double_quant=True,
        )
        mdl = AutoModelForCausalLM.from_pretrained(
            name,
            quantization_config=bnb_cfg,
            device_map=device_map,
            torch_dtype="auto",
            max_memory=max_memory,
            **offload_kwargs
        )

    elif mode == "8bit":
        bnb_cfg = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_enable_fp32_cpu_offload=True,  # Allow CPU offloading
        )
        mdl = AutoModelForCausalLM.from_pretrained(
            name,
            quantization_config=bnb_cfg,
            device_map=device_map,
            torch_dtype="auto",
            max_memory=max_memory,
            **offload_kwargs
        )

    elif mode == "fp16":
        mdl = AutoModelForCausalLM.from_pretrained(
            name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map=device_map,
            max_memory=max_memory,
            **offload_kwargs
        )

    else:  # fp32 (CPU only recommended)
        mdl = AutoModelForCausalLM.from_pretrained(
            name,
            torch_dtype=torch.float32,
            device_map="cpu"
        )

    return tok, mdl

def load_model_safe(name: str, precision: str):
    """Safe model loading"""
    try:
        return load_model(name, precision)
    except Exception as e:
        st.error(f"Model {name} loading failed: {e}")
        return None, None

# Load current model
token, model = load_model(MODEL_NAME, precision)
st.success(f"✅ Model loaded: {MODEL_NAME} ({precision})")

# ==== Evaluation functions ====
def generate_once_with_params(tokenizer, model, messages, temp, top_p_val, max_tokens, rep_penalty):
    """Generate text once with given parameters (apply all 4 parameters)"""

    if tokenizer is None or model is None:
        return "[ERROR] Model not loaded", None

    try:
        # Token ID validation
        vocab_size = model.get_input_embeddings().num_embeddings
        eos_id = tokenizer.eos_token_id
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_id

        if eos_id is not None and eos_id >= vocab_size:
            eos_id = None
        if pad_id is not None and pad_id >= vocab_size:
            pad_id = eos_id

        # Apply all parameters
        gen_kwargs = {
            "max_new_tokens": max_tokens,
            "do_sample": True if temp > 0 else False,
            "temperature": temp,
            "top_p": top_p_val,
            "repetition_penalty": rep_penalty,
        }

        if eos_id is not None:
            gen_kwargs['eos_token_id'] = eos_id
        if pad_id is not None:
            gen_kwargs['pad_token_id'] = pad_id

        # Input processing
        try:
            input_ids = tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
            ).to(model.device)
        except Exception:
            joined = "".join([f"User: {m['content']}\n" if m['role'] == 'user'
                             else f"Assistant: {m['content']}\n" for m in messages])
            joined += "Assistant:"
            input_ids = tokenizer(joined, return_tensors="pt").input_ids.to(model.device)

        attention_mask = torch.ones_like(input_ids)

        # Generate
        start_time = time.time()
        with torch.no_grad():
            gen_ids = model.generate(input_ids, attention_mask=attention_mask, **gen_kwargs)
        latency = time.time() - start_time

        # Decode
        output = tokenizer.decode(
            gen_ids[0][input_ids.shape[1]:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        ).strip()

        if not output:
            output = "No response generated."

        return output, latency

    except Exception as e:
        return f"[ERROR] {str(e)}", None

def comprehensive_score(prompt: str, output: str):
    """Compute comprehensive quality score"""
    if output.startswith("[ERROR]"):
        return 0.0

    scores = {
        'length': min(len(output) / 100, 1.0),
        'completeness': 0.0 if "[ERROR]" in output else 1.0,
        'korean_ratio': len(re.findall(r'[가-힣]', output)) / max(len(output), 1),
        'content_quality': 0.5
    }

    # Specialized scoring
    if "json" in prompt.lower():
        try:
            json.loads(output)
            scores['content_quality'] = 1.0
        except:
            if "[" in output or "{" in output:
                scores['content_quality'] = 0.7
    elif any(word in prompt for word in ["계산", "수학", "×", "+"]):
        if re.search(r'\d+', output):
            scores['content_quality'] = 0.8
    elif "창의" in prompt or "이야기" in prompt:
        sentences = len(output.split('.'))
        if sentences >= 3:
            scores['content_quality'] = 0.9

    weights = {'length': 0.2, 'completeness': 0.3, 'korean_ratio': 0.2, 'content_quality': 0.3}
    final_score = sum(scores[k] * weights[k] for k in weights)
    return final_score

def create_benchmark_prompts():
    """Create benchmark prompts"""
    benchmark_prompts = []
    for category, prompts in EVALUATION_PROMPTS.items():
        benchmark_prompts.extend([{"prompt": p, "category": category} for p in prompts])
    return benchmark_prompts

# ==== Simple chat interface ====
if "messages" not in st.session_state:
    st.session_state.messages = []

st.markdown("### 💬 Quick Test")
if prompt := st.chat_input("Test with the current parameters..."):
    messages = [{"role": "user", "content": prompt}]
    with st.spinner("Generating..."):
        output, latency = generate_once_with_params(
            token, model, messages, temperature, top_p, max_new_tokens, repetition_penalty
        )
    col1, col2 = st.columns([3, 1])
    with col1:
        st.write(f"**Response**: {output}")
    with col2:
        score = comprehensive_score(prompt, output)
        st.metric("Quality Score", f"{score:.3f}", f"{latency:.2f}s")

# ==== Evaluation run ====
if eval_mode and run_eval:
    if use_benchmark:
        benchmark_prompts = create_benchmark_prompts()
        prompts = [item["prompt"] for item in benchmark_prompts]
        categories = [item["category"] for item in benchmark_prompts]
    else:
        prompts = [p.strip() for p in eval_prompts_text.split("\n") if p.strip()]
        categories = ["custom"] * len(prompts)

    if not prompts:
        st.warning("Please enter prompts.")
    else:
        rows = []
        progress_bar = st.progress(0.0)
        status_text = st.empty()

        if baseline_mode:
            # 🧱 Baseline: single model, deterministic decoding
            status_text.info(f"🧱 Baseline: {MODEL_NAME} (do_sample=False, T=0.0, P=1.0, R=1.1, M={baseline_max_tokens})")
            total = len(prompts)
            done = 0

            for i, p in enumerate(prompts):
                messages = [{"role": "user", "content": p}]
                out, lat = generate_once_with_params(
                    token, model, messages,
                    temp=0.0, top_p_val=1.0, max_tokens=baseline_max_tokens, rep_penalty=1.10
                )
                rows.append({
                    "model": MODEL_NAME,
                    "do_sample": False,
                    "temperature": 0.0,
                    "top_p": 1.0,
                    "max_new_tokens": baseline_max_tokens,
                    "repetition_penalty": 1.10,
                    "category": categories[i] if i < len(categories) else "custom",
                    "prompt": p,
                    "output": out,
                    "latency_sec": round(lat, 3) if lat is not None else None,
                    "quality_score": comprehensive_score(p, out),
                })
                done += 1
                progress_bar.progress(done / total)

        elif param_test_mode:
            # 🔁 RAG sweep presets (fixed set)
            status_text.info("🔄 RAG sweep presets: T=[0.2,0.3,0.5,0.7], P=[0.8,0.9,0.95], R=[1.05,1.1,1.15], M=[128,256]")
            test_temps = [0.2, 0.3, 0.5, 0.7]
            test_top_ps = [0.8, 0.9, 0.95]
            test_max_tokens = [128, 256]
            test_rep_penalties = [1.05, 1.10, 1.15]

            param_combinations = [
                (t, p, m, r)
                for t in test_temps
                for p in test_top_ps
                for m in test_max_tokens
                for r in test_rep_penalties
            ]
            st.info(f"📊 Total {len(param_combinations)} parameter combinations to test")

            total = len(param_combinations) * len(prompts)
            done = 0

            for temp_val, top_p_val, max_tok_val, rep_pen_val in param_combinations:
                status_text.info(f"🎛️ Testing: T={temp_val}, P={top_p_val}, M={max_tok_val}, R={rep_pen_val}")
                for i, p in enumerate(prompts):
                    messages = [{"role": "user", "content": p}]
                    try:
                        out, lat = generate_once_with_params(
                            token, model, messages, temp_val, top_p_val, max_tok_val, rep_pen_val
                        )
                    except Exception as e:
                        out, lat = f"[ERROR] {str(e)}", None

                    rows.append({
                        "model": MODEL_NAME,
                        "do_sample": True,
                        "temperature": temp_val,
                        "top_p": top_p_val,
                        "max_new_tokens": max_tok_val,
                        "repetition_penalty": rep_pen_val,
                        "category": categories[i] if i < len(categories) else "custom",
                        "prompt": p,
                        "output": out,
                        "latency_sec": round(lat, 3) if lat is not None else None,
                        "quality_score": comprehensive_score(p, out),
                    })
                    done += 1
                    progress_bar.progress(done / total)

        else:
            # 🔁 Model comparison mode
            models_to_test = [
                "yanolja/EEVE-Korean-2.8B-v1.0",
                "nlpai-lab/KULLM3",
                "EleutherAI/polyglot-ko-5.8b",
            ]
            status_text.info("🔁 Model comparison mode (may take a long time)")
            total = len(models_to_test) * len(prompts)
            done = 0

            for mname in models_to_test:
                status_text.info(f"🔄 Testing model: {mname}")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                try:
                    if mname == MODEL_NAME:
                        tok_i, mdl_i = token, model
                    else:
                        tok_i, mdl_i = load_model_safe(mname, precision)
                    if tok_i is None or mdl_i is None:
                        raise Exception("Model loading failed")
                    for i, p in enumerate(prompts):
                        messages = [{"role": "user", "content": p}]
                        try:
                            out, lat = generate_once_with_params(
                                tok_i, mdl_i, messages, temperature, top_p, max_new_tokens, repetition_penalty
                            )
                        except Exception as e:
                            out, lat = f"[ERROR] {str(e)}", None

                        rows.append({
                            "model": mname,
                            "do_sample": True if temperature > 0 else False,
                            "temperature": temperature,
                            "top_p": top_p,
                            "max_new_tokens": max_new_tokens,
                            "repetition_penalty": repetition_penalty,
                            "category": categories[i] if i < len(categories) else "custom",
                            "prompt": p,
                            "output": out,
                            "latency_sec": round(lat, 3) if lat is not None else None,
                            "quality_score": comprehensive_score(p, out),
                        })
                        done += 1
                        progress_bar.progress(done / total)
                except Exception as e:
                    st.error(f"Model {mname} processing failed: {e}")
                    for p in prompts:
                        rows.append({
                            "model": mname,
                            "temperature": temperature,
                            "top_p": top_p,
                            "max_new_tokens": max_new_tokens,
                            "repetition_penalty": repetition_penalty,
                            "category": "error",
                            "prompt": p,
                            "output": f"[MODEL ERROR] {str(e)}",
                            "latency_sec": None,
                            "quality_score": 0.0,
                        })
                        done += 1
                        progress_bar.progress(done / total)

        # Results analysis
        status_text.success("✅ Evaluation completed!")

        if rows:
            df = pd.DataFrame(rows)

            # Results table
            st.markdown("### 📊 Evaluation Results")
            st.dataframe(df, use_container_width=True)

            # CSV download
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 Download Results (CSV)",
                data=csv,
                file_name=f"performance_test_{time.strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )

            # Performance summary
            st.markdown("### 🏆 Performance Summary")

            if param_test_mode:
                # Parameter-wise performance analysis
                param_summary = df.groupby(['temperature', 'top_p', 'max_new_tokens', 'repetition_penalty']).agg(
                    avg_quality=('quality_score', 'mean'),
                    avg_latency=('latency_sec', 'mean'),
                    success_rate=('quality_score', lambda x: (x > 0.5).mean())
                ).reset_index()

                param_summary = param_summary.sort_values('avg_quality', ascending=False)
                st.dataframe(param_summary, use_container_width=True)

                # Recommend best parameter set
                best_params = param_summary.iloc[0]
                st.success(f"🎯 **Best Parameter Combination**")
                st.write(f"- Temperature: {best_params['temperature']}")
                st.write(f"- Top-p: {best_params['top_p']}")
                st.write(f"- Max Tokens: {best_params['max_new_tokens']}")
                st.write(f"- Repetition Penalty: {best_params['repetition_penalty']}")
                st.write(f"- Avg Quality Score: {best_params['avg_quality']:.3f}")
                st.write(f"- Avg Latency: {best_params['avg_latency']:.2f}s")

            else:
                # Model-wise performance analysis
                model_summary = df.groupby('model').agg(
                    avg_quality=('quality_score', 'mean'),
                    avg_latency=('latency_sec', 'mean'),
                    success_rate=('quality_score', lambda x: (x > 0.5).mean()),
                    error_rate=('output', lambda x: x.str.startswith('[ERROR]').mean())
                ).reset_index()

                model_summary = model_summary.sort_values('avg_quality', ascending=False)
                st.dataframe(model_summary, use_container_width=True)

                # Best model
                best_model = model_summary.iloc[0]
                st.success(f"🏆 **Best Model**: {best_model['model']}")
                st.write(f"- Avg Quality Score: {best_model['avg_quality']:.3f}")
                st.write(f"- Success Rate: {best_model['success_rate']:.1%}")
                st.write(f"- Avg Latency: {best_model['avg_latency']:.2f}s")

            # Category-wise performance
            if 'category' in df.columns:
                st.markdown("### 📈 Category-wise Performance")
                category_summary = df.groupby('category').agg(
                    avg_quality=('quality_score', 'mean'),
                    avg_latency=('latency_sec', 'mean'),
                    count=('prompt', 'count')
                ).reset_index()

                category_summary = category_summary.sort_values('avg_quality', ascending=False)
                st.dataframe(category_summary, use_container_width=True)

        else:
            st.error("No evaluation results available.")

# Reset button
with st.sidebar:
    st.markdown("---")
    if st.button("🔄 Reset All"):
        st.session_state.clear()
        st.rerun()
