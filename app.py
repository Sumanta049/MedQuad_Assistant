import os
import re
import torch
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Configuration
BASE_MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"
ADAPTER_ID = "Lucifer049/medquad_assistant_v1"

SYSTEM_PROMPT = (
    "You are a helpful and accurate medical assistant. "
    "Answer in a clear, structured format. "
    "For every response: start with 1 brief context sentence, then add 'Key points:' followed by 4-7 bullet points. "
    "If the topic can involve danger signs, add 'See a doctor urgently if:' with bullet points. "
    "Keep statements medically grounded, practical, and non-alarmist. "
    "Avoid repetition and vague filler text. "
    "End with a brief reminder that this is educational information and not a diagnosis."
)

# Generation parameters
MAX_NEW_TOKENS = 320
TEMPERATURE = 0.25
TOP_P = 0.85
REPETITION_PENALTY = 1.1
NO_REPEAT_NGRAM_SIZE = 4

# Model Loading
_tokenizer = None
_model = None


def _get_hf_token() -> str:
    # Get HuggingFace token from environment variables.
    return (os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN") or "").strip()


def _load_model():
    # Load the base model and attach the fine-tuned adapter.
    global _tokenizer, _model

    if _tokenizer is not None and _model is not None:
        return _tokenizer, _model

    # Validate CUDA availability
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. This application requires an NVIDIA GPU with CUDA support."
        )

    hf_token = _get_hf_token() or None
    
    # Determine optimal dtype based on GPU capability
    capability = torch.cuda.get_device_capability()
    use_bf16 = capability[0] >= 8  # Ampere+ supports bfloat16 efficiently

    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"Using {'bfloat16' if use_bf16 else 'float16'} compute dtype")

    # 4-bit quantization config for memory efficiency
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16 if use_bf16 else torch.float16,
    )

    # Load tokenizer
    print(f"Loading tokenizer from {BASE_MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load base model with quantization
    print(f"Loading base model from {BASE_MODEL_ID}...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        token=hf_token,
        device_map="auto",
        quantization_config=quant_config,
    )

    # Load fine-tuned adapter
    print(f"Loading adapter from {ADAPTER_ID}...")
    model = PeftModel.from_pretrained(base_model, ADAPTER_ID, token=hf_token)
    model.eval()

    memory_mb = model.get_memory_footprint() / 1e6
    print(f"Model loaded. Memory footprint: {memory_mb:.1f} MB")

    _tokenizer = tokenizer
    _model = model
    return _tokenizer, _model


def _ensure_bulleted_response(text: str) -> str:
    """Normalize model output into point-wise format when needed."""
    cleaned = (text or "").strip()
    if not cleaned:
        return cleaned

    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
    has_points = any(
        line.startswith("- ")
        or line.startswith("* ")
        or bool(re.match(r"^\d+\.\s", line))
        for line in lines
    )
    if has_points:
        return cleaned

    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", cleaned) if s.strip()]
    if not sentences:
        return cleaned

    if len(sentences) == 1:
        return f"Key points:\n- {sentences[0]}"

    intro = sentences[0]
    points = sentences[1:]

    formatted = [intro, "", "Key points:"]
    formatted.extend([f"- {point}" for point in points])
    return "\n".join(formatted)


# Inference
def respond(message: str, history: list) -> str:
    """
    Generate a response to a medical query.
    
    Args:
        message: The user's current message
        history: List of previous (user, assistant) message tuples
        
    Returns:
        The assistant's response text
    """
    try:
        tokenizer, model = _load_model()

        # Build conversation history in chat format
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        
        for turn in history:
            if isinstance(turn, (list, tuple)) and len(turn) == 2:
                user_msg, assistant_msg = turn
                if user_msg:
                    messages.append({"role": "user", "content": str(user_msg)})
                if assistant_msg:
                    messages.append({"role": "assistant", "content": str(assistant_msg)})

        messages.append({"role": "user", "content": message})

        # Apply Llama 3.2 chat template
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        # Generate response
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                do_sample=True,
                repetition_penalty=REPETITION_PENALTY,
                no_repeat_ngram_size=NO_REPEAT_NGRAM_SIZE,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id,
            )

        # Extract only the generated tokens (not the prompt)
        generated_tokens = outputs[0][inputs["input_ids"].shape[-1]:]
        response = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        response = _ensure_bulleted_response(response)
        
        return response

    except RuntimeError as e:
        if "CUDA" in str(e):
            return "**Error**: CUDA is not available. This app requires an NVIDIA GPU."
        raise
    except Exception as e:
        error_msg = str(e)
        if "401" in error_msg or "unauthorized" in error_msg.lower():
            return "**Authentication Error**: Please set your HuggingFace token: `export HF_TOKEN='your_token'`"
        return f"**Error**: {error_msg}"


# Gradio Interface
MEDICAL_DISCLAIMER = """
**Medical Disclaimer**: This chatbot is for **educational purposes only**. 
It is NOT a substitute for professional medical advice, diagnosis, or treatment. 
Always consult a qualified healthcare provider for medical concerns.
"""

demo = gr.ChatInterface(
    fn=respond,
    title="MedQuad Assistant",
    description=MEDICAL_DISCLAIMER,
    examples=[
        "What are the common symptoms of Type 2 Diabetes?",
        "How is asthma typically treated?",
        "What causes high blood pressure?",
        "What are the signs of dehydration?",
    ],
    theme="ocean",
)


if __name__ == "__main__":
    print("=" * 60)
    print("MedQuad Assistant - Medical Q&A Chatbot")
    print("=" * 60)
    
    # Check for HuggingFace token
    if not _get_hf_token():
        print("\n Warning: HuggingFace token not set!")
        print("   Set it with: export HF_TOKEN='your_token'")
        print("   Get a token at: https://huggingface.co/settings/tokens\n")

    # Check CUDA availability
    if torch.cuda.is_available():
        print(f"CUDA available: {torch.cuda.get_device_name()}")
    else:
        print("CUDA not available - this app requires an NVIDIA GPU")
        exit(1)
    
    print(f"\nBase model: {BASE_MODEL_ID}")
    print(f"Adapter: {ADAPTER_ID}")
    print("4-bit quantization enabled")
    print("\nStarting Gradio server...")
    print("=" * 60)

    demo.launch(share=True)
    