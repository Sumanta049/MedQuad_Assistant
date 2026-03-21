"""
Medical Q&A Chatbot - Gradio Interface
Fine-tuned Llama 3.2 model for medical question answering
"""

import gradio as gr
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# Configuration
MODEL_ID = "Lucifer049/med_chatbot_finetuned"  # Your HuggingFace model
BASE_MODEL_ID = os.getenv("BASE_MODEL_ID", "meta-llama/Llama-3.2-3B-Instruct")
MAX_NEW_TOKENS = 512
TEMPERATURE = 0.7
TOP_P = 0.9

# Medical disclaimer
DISCLAIMER = """
**Medical Disclaimer**: This chatbot is for educational and informational purposes only. 
It is NOT a substitute for professional medical advice, diagnosis, or treatment. 
Always consult a qualified healthcare provider for medical concerns.
"""

def load_model():
    """Load model for inference.

    Supports either:
    1) Full merged checkpoints (MODEL_ID is directly loadable), or
    2) Adapter-only repos (falls back to BASE_MODEL_ID + PEFT adapter).
    """
    print("Loading model...")
    
    # Quantization config for efficient inference
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=False,
    )
    
    # Try loading as a full/merged model first.
    try:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    except ValueError as err:
        # Adapter-only repos usually fail here with missing `model_type` in config.json.
        if "model_type" not in str(err):
            raise

        print("Detected adapter-only repo. Loading base model + LoRA adapter...")
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_ID,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(base_model, MODEL_ID)

        # Prefer adapter tokenizer if present, otherwise use base tokenizer.
        try:
            tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
        except Exception:
            tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)

    tokenizer.pad_token = tokenizer.eos_token
    
    print("Model loaded successfully!")
    return model, tokenizer

# Load model at startup
model, tokenizer = load_model()

def format_chat_history(history: list) -> str:
    """Format conversation history for the model."""
    messages = [{"role": "system", "content": "You are a helpful and accurate medical assistant."}]
    
    for user_msg, assistant_msg in history:
        messages.append({"role": "user", "content": user_msg})
        if assistant_msg:
            messages.append({"role": "assistant", "content": assistant_msg})
    
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

def generate_response(message: str, history: list) -> str:
    """Generate a response to the user's medical question."""
    # Add current message to history for formatting
    history_with_current = history + [[message, None]]
    prompt = format_chat_history(history_with_current)
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    # Decode response (only the new tokens)
    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    )
    
    return response.strip()

# Example questions
EXAMPLES = [
    "What are the common symptoms of diabetes?",
    "How can I manage high blood pressure naturally?",
    "What causes lower back pain?",
    "What are the early signs of heart disease?",
    "How do I know if I have a vitamin deficiency?",
]

# Build Gradio interface
with gr.Blocks(
    title="MedQuad Assistant - Medical Q&A Chatbot",
    theme=gr.themes.Soft(primary_hue="blue"),
) as demo:
    gr.Markdown("# MedQuad Assistant - Medical Q&A Chatbot")
    gr.Markdown(DISCLAIMER)
    gr.Markdown(
        """
        This chatbot is powered by a **Llama 3.2-3B** model fine-tuned on the 
        [MedQuad](https://huggingface.co/datasets/keivalya/MedQuad-MedicalQnADataset) 
        medical Q&A dataset (~16,000 medical question-answer pairs).
        """
    )
    
    chatbot = gr.Chatbot(
        label="Chat",
        height=450,
        bubble_full_width=False,
    )
    
    msg = gr.Textbox(
        label="Your Question",
        placeholder="Ask a medical question...",
        lines=2,
    )
    
    with gr.Row():
        submit = gr.Button("Send", variant="primary")
        clear = gr.Button("Clear Chat")
    
    gr.Markdown("### 💡 Example Questions")
    gr.Examples(
        examples=EXAMPLES,
        inputs=msg,
    )
    
    # Event handlers
    def respond(message, chat_history):
        response = generate_response(message, chat_history)
        chat_history.append((message, response))
        return "", chat_history
    
    submit.click(respond, [msg, chatbot], [msg, chatbot])
    msg.submit(respond, [msg, chatbot], [msg, chatbot])
    clear.click(lambda: None, None, chatbot, queue=False)
    
    gr.Markdown(
        """
        ---
        **Model**: [Lucifer049/med_chatbot_finetuned](https://huggingface.co/Lucifer049/med_chatbot_finetuned) | 
        **Dataset**: [MedQuad](https://huggingface.co/datasets/keivalya/MedQuad-MedicalQnADataset) |
        **Base Model**: Llama 3.2-3B-Instruct
        """
    )

if __name__ == "__main__":
    demo.queue().launch()
