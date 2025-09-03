import torch
from transformers import pipeline
import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

model_id = "meta-llama/Llama-3.2-1B-Instruct"
mps_device = "mps"
pipe = pipeline(
    "text-generation",
    model=model_id,
    torch_dtype=torch.bfloat16,
    device = mps_device,
)
messages = [
    {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
    {"role": "user", "content": "Who are you?"},
]
outputs = pipe(
    messages,
    max_new_tokens=256,
)
print(outputs[0]["generated_text"][-1])
