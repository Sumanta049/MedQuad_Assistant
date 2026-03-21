# 🏥 Medical Q&A Chatbot

A conversational AI assistant for medical questions, powered by a fine-tuned **Llama 3.2-3B** model.

[![Hugging Face](https://img.shields.io/badge/🤗%20Model-Hugging%20Face-yellow)](https://huggingface.co/Lucifer049/med_chatbot_finetuned)
[![Gradio](https://img.shields.io/badge/Demo-Gradio-orange)](https://huggingface.co/spaces/Lucifer049/med-chatbot)

> ⚠️ **Disclaimer**: This chatbot is for educational purposes only. It is NOT a substitute for professional medical advice.

---

## 🎯 Project Overview

This project demonstrates end-to-end LLM fine-tuning and deployment:

1. **Fine-tuned** Meta's Llama 3.2-3B-Instruct on 16,407 medical Q&A pairs
2. **Optimized** training with QLoRA (4-bit quantization + LoRA adapters)
3. **Deployed** interactive chat interface via Gradio on HuggingFace Spaces

## 🧠 Model Details

| Attribute | Value |
|-----------|-------|
| Base Model | `meta-llama/Llama-3.2-3B-Instruct` |
| Fine-tuned Model | `Lucifer049/med_chatbot_finetuned` |
| Dataset | [MedQuad-MedicalQnADataset](https://huggingface.co/datasets/keivalya/MedQuad-MedicalQnADataset) |
| Training Samples | 16,407 |
| Training Method | QLoRA (4-bit NF4 quantization) |
| LoRA Config | r=64, alpha=16, dropout=0.1 |
| Training Time | ~3 hours (Kaggle T4 GPU) |

## 🚀 Quick Start

### Run Locally

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/med-chatbot.git
cd med-chatbot

# Install dependencies
pip install -r requirements.txt

# Run the app
python app.py
```

### Use the Model Directly

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

model_id = "Lucifer049/med_chatbot_finetuned"

# Load model
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype="auto",
)

# Create pipeline
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Generate response
messages = [
    {"role": "system", "content": "You are a helpful and accurate medical assistant."},
    {"role": "user", "content": "What are the symptoms of diabetes?"}
]

response = pipe(messages, max_new_tokens=256)
print(response[0]["generated_text"][-1]["content"])
```

## 📊 Sample Outputs

### Question: "What are the common symptoms of back pain?"

> Back pain symptoms can vary depending on the underlying cause. Common symptoms include:
> - Muscle ache or stiffness in the lower back
> - Pain that radiates down the leg (sciatica)
> - Limited flexibility or range of motion
> - Difficulty standing straight
> 
> The pain may be constant or may occur only during certain movements...

### Question: "How can I manage high blood pressure naturally?"

> There are several lifestyle modifications that can help manage blood pressure:
> - Reduce sodium intake (less than 2,300mg per day)
> - Maintain a healthy weight
> - Exercise regularly (at least 30 minutes most days)
> - Limit alcohol consumption
> - Manage stress through relaxation techniques
> - Follow the DASH diet (rich in fruits, vegetables, whole grains)...

## 📁 Project Structure

```
med-chatbot/
├── app.py              # Gradio chat interface
├── requirements.txt    # Python dependencies
├── README.md           # Project documentation
└── notebook/
    └── fine_tuning.ipynb  # Training notebook (Kaggle)
```

## 🔧 Technical Stack

- **Model**: Meta Llama 3.2-3B-Instruct
- **Fine-tuning**: Hugging Face TRL (SFTTrainer) + PEFT (LoRA)
- **Quantization**: BitsAndBytes (4-bit NF4)
- **Frontend**: Gradio
- **Deployment**: HuggingFace Spaces
- **Training Platform**: Kaggle (T4 GPU)

## 📈 Training Configuration

```python
# LoRA Configuration
peft_params = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="CAUSAL_LM",
)

# Training Arguments
training_params = SFTConfig(
    output_dir="./results",
    num_train_epochs=1,
    per_device_train_batch_size=1,
    max_length=512,
    logging_steps=25,
    save_steps=500,
)
```

## ⚠️ Limitations

- **Not for medical decisions**: This model should not be used for actual medical diagnosis or treatment
- **Training data bias**: Responses are limited to the scope of the MedQuad dataset
- **Hallucination risk**: Like all LLMs, may generate plausible-sounding but incorrect information
- **No real-time knowledge**: Model knowledge is limited to training data cutoff

## 📝 License

This project is for educational purposes. The base Llama model is subject to Meta's [Llama License](https://ai.meta.com/llama/license/).

## 🙏 Acknowledgments

- [Meta AI](https://ai.meta.com/) for Llama 3.2
- [Hugging Face](https://huggingface.co/) for transformers, TRL, and PEFT libraries
- [MedQuad Dataset](https://huggingface.co/datasets/keivalya/MedQuad-MedicalQnADataset) creators
- [Kaggle](https://kaggle.com/) for free GPU compute

---

**Author**: [Sumanta Mukherjee]  
**Contact**: [sumantamukherjee0049@google.com]
