"""Load the model and generate text for a new prompt."""
import os
import sys
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.ch03_tokenizer import build_vocab, encode, decode
from src.ch09_gpt_model import GPT
from src.ch15_generate_sampling import generate

def main():
    torch.manual_seed(42)
    checkpoint_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "checkpoints", "model_final.pt"
    )
    
    if not os.path.exists(checkpoint_path):
        # Fall back to model.pt if model_final.pt does not exist
        checkpoint_path = os.path.join(
            os.path.dirname(__file__), "..", "..", "checkpoints", "model.pt"
        )
        
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    
    cfg = checkpoint["gpt_cfg"]
    model = GPT(cfg)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    
    text_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "src", "data", "shakespeare.txt"
    )
    with open(text_path, "r", encoding="utf-8") as f:
        text = f.read()
    chars, char_to_id, id_to_char = build_vocab(text)
    
    encode_fn = lambda s: encode(s, char_to_id)
    decode_fn = lambda ids: decode(ids, id_to_char)
    
    prompt = "USER: What is the news?\n"
    print(f"--- Prompt: {repr(prompt)} ---")
    
    output = generate(
        model, prompt, encode_fn, decode_fn, cfg, 
        max_new_tokens=100, temperature=0.8, top_k=40
    )
    print(output)

if __name__ == "__main__":
    main()
