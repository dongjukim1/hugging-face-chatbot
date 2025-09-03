import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

st.set_page_config(page_title="MY TEST CHATBOT", page_icon="😛", layout="centered")
st.title("😛 MY TEST CHATBOT")
st.subheader("Hugging Face CHAT TEMPLATE")

MODEL_NAME = st.sidebar.selectbox(
    "MODEL",
    [
        "Qwen/Qwen2.5-1.5B-Instruct",
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    ]
)

@st.cache_resource
def load_model(name):
    token = AutoTokenizer.from_pretrained(name, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float32, device_map="cpu")
    return token, model

token, model = load_model(MODEL_NAME)
st.success("✅ Model loaded successfully")

if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.write(m["content"])

if prompt := st.chat_input("Enter your message..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Generating.."):
            chat = []
            for m in st.session_state.messages:
                if m["role"] == "user":
                    chat.append({"role": "user", "content": m["content"]})
                else:
                    chat.append({"role": "assistant", "content": m["content"]})

            input_ids = token.apply_chat_template(
                chat,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt"
            )

            input_ids = input_ids.to(model.device)

            gen_ids = model.generate(
                input_ids,
                max_new_tokens=256,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1,
                eos_token_id=token.eos_token_id,
                pad_token_id=token.eos_token_id,
            )

            output = token.decode(gen_ids[0][input_ids.shape[1]:], skip_special_tokens=True).strip()
            if not output:
                output = "Sorry, could you please say that again?"
            st.write(output)
            st.session_state.messages.append({"role": "assistant", "content": output})

with st.sidebar:
    st.markdown("---")
    if st.button("Reset conversation"):
        st.session_state.messages = []
        st.rerun()
