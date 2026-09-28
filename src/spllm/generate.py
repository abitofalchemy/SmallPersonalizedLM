"""Text generation with temperature and top-k sampling (Raschka, ch. 5)."""

import tiktoken
import torch


def text_to_token_ids(text: str, tokenizer) -> torch.Tensor:
    encoded = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    return torch.tensor(encoded).unsqueeze(0)


def token_ids_to_text(token_ids: torch.Tensor, tokenizer) -> str:
    return tokenizer.decode(token_ids.squeeze(0).tolist())


@torch.no_grad()
def generate(model, idx, max_new_tokens, context_size, temperature=0.0, top_k=None, eos_id=None):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        logits = model(idx_cond)[:, -1, :]

        if top_k is not None:
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1:]
            logits = torch.where(logits < min_val, torch.full_like(logits, -torch.inf), logits)

        if temperature > 0.0:
            probs = torch.softmax(logits / temperature, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)

        if eos_id is not None and (idx_next == eos_id).all():
            break
        idx = torch.cat((idx, idx_next), dim=1)
    return idx


def generate_text(model, prompt: str, max_new_tokens=50, temperature=0.0, top_k=None) -> str:
    tokenizer = tiktoken.get_encoding("gpt2")
    device = next(model.parameters()).device
    idx = text_to_token_ids(prompt, tokenizer).to(device)
    model.eval()
    out = generate(
        model,
        idx,
        max_new_tokens=max_new_tokens,
        context_size=model.cfg.context_length,
        temperature=temperature,
        top_k=top_k,
    )
    return token_ids_to_text(out.cpu(), tokenizer)
