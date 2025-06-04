import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from peft import PeftModel

# Load RAG module
import utils.rag_system as rag_module

# ==== Device and Model Loading ====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def build_chat_prompt(user_message):
    return (
        "<|system|>\n"
        "You are a helpful assistant knowledgeable in Shakespearean literature.\n"
        "<|user|>\n"
        f"{user_message}\n"
        "<|assistant|>\n"
    )

def is_analytical_question(query: str) -> bool:
    lowered = query.lower().strip()
    starters = ["why", "what causes", "what motivates", "how does", "how do", "explain", "in what way", "discuss", "analyze", "interpret"]
    return any(lowered.startswith(start) for start in starters)

def route_query(query: str, rag, generator):
    top_types, scores = rag.classify_query(query)
    max_score = max(scores.values())

    if is_analytical_question(query):
        return run_generator(query, generator), "SLM Generator (interpretive override)"
    elif max_score < 6.0:
        return run_generator(query, generator), "SLM Generator (low score fallback)"
    else:
        results = rag.search(query=query, k=3, chunk_types=top_types, rerank=True)
        return (results[0].content.strip() if results else "No relevant answer found."), "RAG"

def run_generator(query: str, generator):
    prompt = build_chat_prompt(query)
    output = generator(prompt, max_new_tokens=200, do_sample=True, top_k=50, top_p=0.95, temperature=0.7)[0]["generated_text"]
    response_start = output.find("<|assistant|>\n")
    if response_start != -1:
        return output[response_start + len("<|assistant|>\n"):].strip()
    return output.strip()

# ==== CLI App ====
def run_cli_chat():
    # Intro Banner
    print("\n" + "╔" + "═" * 62 + "╗")
    print("║{:^60}║".format("🎭 SHAKESPEARE CHATBOT CLI 🎭"))
    print("║{:^62}║".format('"To chat, or not to chat..."'))
    print("╚" + "═" * 62 + "╝")

    # Load models
    print("🛠️  Loading models and RAG system...")
    rag = rag_module.ShakespeareRAGSystem(index_dir="retrieval/index")

    base_model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0", torch_dtype=torch.float32)
    model = PeftModel.from_pretrained(base_model, "models/gpt-neo-03").eval().to(device)
    tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0 if device.type == "cuda" else -1)

    # Main loop
    while True:
        try:
            user_input = input("\n🧑‍🎓 You (Enter your Shakespeare question or type 'quit' to exit): ").strip()
            if user_input.lower() in ["exit", "quit", "bye"]:
                print("\n🎭 Goodbye! 'Parting is such sweet sorrow...'\n")
                break
            if not user_input:
                continue

            print("🤖 Thinking...")
            response, source = route_query(user_input, rag, generator)
            print(f"🤖 Bot ({source}): {response}")

        except KeyboardInterrupt:
            print("\n🎭 Interrupted. Farewell!\n")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            continue

if __name__ == "__main__":
    run_cli_chat()