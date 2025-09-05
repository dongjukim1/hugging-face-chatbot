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
        help="Use 4bit/8bit for limited memory, fp16/fp32 if sufficient"
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

    # Baseline Max Tokens (options: 128/256)
    baseline_max_tokens = st.selectbox(
        "Baseline Max Tokens", [128, 256], index=0,
        help="128 recommended for short QA. Use 256 if needed."
    )

    eval_mode = st.checkbox("Enable evaluation mode", value=False)

    # Sweep mode toggle
    param_test_mode = st.checkbox("Parameter sweep test (RAG presets)", value=False,
                                  help="do_sample=True with RAG presets (temp/topP/repetition/max tokens=128·256)")

    # (Optional) Custom prompt input
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
        # Set GPU and CPU memory limits
        max_memory = {0: "7GiB", "cpu": "32GiB"}

    offload_kwargs = dict(
        offload_folder="offload",      # Offloading folder
        offload_state_dict=True,       # Also offload state dict
    )
