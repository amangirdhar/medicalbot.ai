import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login
HF_TOKEN = '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'
login(token=HF_TOKEN)

model_id = "Khushiee/medical-gemma2b-chatbot"
tokenizer = AutoTokenizer.from_pretrained(model_id, use_auth_token=HF_TOKEN)
model = AutoModelForCausalLM.from_pretrained(model_id, use_auth_token=HF_TOKEN)
device = "cuda:0" if torch.cuda.is_available() else "cpu"
model.to(device)
text = "question: Suggest some medication for the headache;"
inputs = tokenizer(text, return_tensors="pt").to(device)
outputs = model.generate(**inputs, max_new_tokens=20)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
