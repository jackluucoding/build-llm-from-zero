"""Generate text with a very low temperature to simulate greedy decoding."""
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
        os.path.dirname(__file__), "..", "..", "checkpoints", "model.pt"
    )
    
    if not os.path.exists(checkpoint_path):
        print("Checkpoint not found.")
        return
        
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
    
    prompt = "JULIET:\n"
    print(f"--- Prompt: {repr(prompt)} ---")
    print("Generating with T=0.1...")
    
    # Using the generation function that takes temperature and top_k
    output = generate(
        model, prompt, encode_fn, decode_fn, cfg, 
        max_new_tokens=40, temperature=0.1, top_k=40
    )
    print(output + "...\n")

if __name__ == "__main__":
    main()
