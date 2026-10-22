"""
Fair-decoding protocol from the original repo: NLLB's beam search favors shorter
hypotheses, so comparing a trained model against vanilla under one fixed decoding
setting can be biased (the two systems are not equally sensitive to it). Each system --
vanilla and trained, independently -- gets its own best decoding configuration: three
settings are scored on a small FLORES-200 dev subset, the best-scoring setting is fixed,
and that fixed setting is applied once to the held-out test set. The paper itself does
not describe this protocol; its Table 7 baseline matches the fixed "default(beam5,lp1.0)"
setting instead (see REPRODUCTION.md).
"""
import torch

GRID = {
    "default(beam5,lp1.0)": dict(num_beams=5, length_penalty=1.0, max_new_tokens=128),
    "lp2.0,cap256": dict(num_beams=5, length_penalty=2.0, max_new_tokens=256),
    "lp3.0,cap256,beam8": dict(num_beams=8, length_penalty=3.0, max_new_tokens=256),
}


def translate(model, tokenizer, tgt_token_id, srcs, generate_kwargs, batch_size=16, device="cuda"):
    """Batched beam-search translation under one fixed decoding setting."""
    hyps = []
    for i in range(0, len(srcs), batch_size):
        batch = tokenizer(srcs[i:i + batch_size], return_tensors="pt", padding=True,
                           truncation=True, max_length=256).to(device)
        with torch.no_grad():
            gen_ids = model.generate(**batch, forced_bos_token_id=tgt_token_id, **generate_kwargs)
        hyps += tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
    return hyps


def dev_tune(model, tokenizer, tgt_token_id, dev_srcs, dev_refs, batch_size=16, device="cuda"):
    """Scores every setting in GRID on a dev subset, returns the name of the winner."""
    from metrics.chrf import chrf_corpus

    best_name, best_score = None, -1.0
    for name, kwargs in GRID.items():
        hyps = translate(model, tokenizer, tgt_token_id, dev_srcs, kwargs, batch_size=batch_size, device=device)
        score = chrf_corpus(hyps, dev_refs)
        if score > best_score:
            best_name, best_score = name, score
    return best_name


def fair_decode(model, tokenizer, tgt_token_id, dev_srcs, dev_refs, test_srcs, batch_size=16, device="cuda"):
    """
    Full fair-decoding protocol for one system: dev-tune the 3-setting grid, then apply
    the winning setting once to the test set. Returns (test_hyps, chosen_setting_name).
    """
    setting_name = dev_tune(model, tokenizer, tgt_token_id, dev_srcs, dev_refs, batch_size, device=device)
    hyps = translate(model, tokenizer, tgt_token_id, test_srcs, GRID[setting_name], batch_size, device=device)
    return hyps, setting_name
