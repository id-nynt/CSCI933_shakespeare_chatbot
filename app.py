import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from peft import PeftModel
import utils.rag_system as rag_module

# === Streamlit UI ===
st.set_page_config(page_title="Shakespeare Chatbot", layout="wide")
st.title("🎭 Shakespeare Chatbot")

# === Device detection ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Load RAG System ===
@st.cache_resource
def load_rag():
    return rag_module.ShakespeareRAGSystem(index_dir="retrieval/index")

# === Load Generator ===
@st.cache_resource
def load_generator():
    base_model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    model = PeftModel.from_pretrained(base_model, "models/tini_llama_01").eval().to(device)
    tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    gen = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0 if device.type == "cuda" else -1)
    return gen, tokenizer

rag = load_rag()
generator, tokenizer = load_generator()

# === Prompt formatter ===
def build_chat_prompt(user_message):
    return (
        "<|system|>\n"
        "You are a helpful assistant knowledgeable in Shakespearean literature.\n"
        "<|user|>\n"
        f"{user_message}\n"
        "<|assistant|>\n"
    )

# === Override checker ===
def is_analytical_question(query: str) -> bool:
    lowered = query.lower().strip()
    starters = ["why", "what causes", "what motivates", "how does", "how do", "explain", "in what way", "discuss", "analyze", "interpret"]
    return any(lowered.startswith(start) for start in starters)

# === Routing logic ===
def route_query(query: str):
    top_types, scores = rag.classify_query(query)
    max_score = max(scores.values())

    if is_analytical_question(query):
        source = "SLM"
        response = run_generator(query)
    elif max_score < 6.0:
        source = "SLM"
        response = run_generator(query)
    else:
        source = "RAG"
        results = rag.search(query=query, k=3, chunk_types=top_types, rerank=True)
        response = results[0].content.strip() if results else "No relevant answer found."

    return response, source

# === Run generator ===
def run_generator(query: str):
    prompt = build_chat_prompt(query)
    output = generator(prompt, max_new_tokens=200, do_sample=True, top_k=50, top_p=0.95, temperature=0.7)[0]["generated_text"]
    response_start = output.find("<|assistant|>\n")
    if response_start != -1:
        return output[response_start + len("<|assistant|>\n"):].strip()
    return output.strip()


# Session state for history
if "history" not in st.session_state:
    st.session_state.history = []

# User input
query = st.text_input("🎤 Enter your Shakespeare question (or type 'quit' to exit):")

if query.strip().lower() == "quit":
    st.session_state.quit = True

if "quit" in st.session_state and st.session_state.quit:
    st.markdown("🎭 **Goodbye!** 'Parting is such sweet sorrow...'\n")
    st.stop()
    
if st.button("Submit") and query.strip():
    with st.spinner("Thinking..."):
        response, source = route_query(query.strip())
        st.session_state.history.append({"query": query.strip(), "response": response, "source": source})

# Chat history
if st.session_state.history:
    for entry in reversed(st.session_state.history):
        with st.container():
            st.markdown(f"**🧑‍🎓 You:** {entry['query']}")
            st.markdown(f"**🤖 Bot ({entry['source']}):** {entry['response']}")
            st.code(entry['response'], language="markdown")
            st.button("📋 Copy", key=f"copy_{entry['query']}", on_click=st.write, args=("Copied!",))
            st.markdown("---")