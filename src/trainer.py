"""
The GRPO training step and training loop. See the paper's Methodology section for the
full derivation; this module is the code that produced every reported number.
"""
import csv
import os

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm import tqdm

from metrics.chrf import chrf_corpus
from src.models import DEVICE

CLIP_LOW, CLIP_HIGH = 0.8, 1.2


class TrainingLogger:
    def __init__(self, filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        write_header = not os.path.exists(filepath) or os.path.getsize(filepath) == 0
        self._file = open(filepath, "a", newline="")
        self._writer = csv.writer(self._file)
        if write_header:
            self._writer.writerow(["lang", "step", "loss", "reward_raw", "kl", "chrf", "temperature"])

    def log(self, lang, step, loss, reward_raw, kl, chrf=None, temp=None):
        self._writer.writerow([lang, step, round(loss, 6), round(reward_raw, 6), round(kl, 6),
                                round(chrf, 4) if chrf is not None else "",
                                round(temp, 4) if temp is not None else ""])
        self._file.flush()

    def close(self):
        self._file.close()


def get_log_probs(model, tokenizer, input_ids, attention_mask, gen_ids):
    """Mean per-token log-probability (length-normalized) of gen_ids under `model`."""
    outputs = model(input_ids=input_ids, attention_mask=attention_mask,
                     decoder_input_ids=gen_ids[:, :-1], return_dict=True)
    log_probs = F.log_softmax(outputs.logits, dim=-1)
    actual_ids = gen_ids[:, 1:].unsqueeze(-1)
    token_log_probs = torch.gather(log_probs, -1, actual_ids).squeeze(-1)
    mask = gen_ids[:, 1:] != tokenizer.pad_token_id
    return (token_log_probs * mask).sum(dim=1) / mask.sum(dim=1)


def train_step(actor, tokenizer, rewarder, optimizer, src_text, lang, temp_controller,
               k=12, max_new_tokens=64, kl_beta=0.001):
    """
    One GRPO update: sample K hypotheses, score them, take a single clipped-surrogate +
    KL-penalized gradient step.

    Note on the ratio: ref_log_probs is computed with the LoRA adapter disabled (the
    frozen pretrained base); curr_log_probs with the adapter enabled, the same weights
    that did the sampling (one gradient step per batch, no stale-policy reuse). So the
    ratio measures adapter-vs-base divergence, not an old-vs-new-policy importance
    ratio -- it is not trivially 1, and the clip range is not inert.

    Note on the KL term: as implemented, this computes exp(delta) - delta - 1 with
    delta = log(pi_theta) - log(pi_ref). The unbiased k3 estimator of
    KL(pi_theta || pi_ref) for samples drawn from pi_theta is exp(-delta) + delta - 1
    (Schulman, http://joschu.net/blog/kl-approx.html) -- the sign on delta inside the
    exponential here does not match that, so this is a biased KL proxy, not the textbook
    k3 estimator. Every reported result was trained with this exact formula; it is kept
    as-is for faithful reproduction rather than silently "corrected" after the fact.
    """
    tokenizer.src_lang = "eng_Latn"
    tokenizer.tgt_lang = lang
    tgt_token_id = tokenizer.convert_tokens_to_ids(lang)
    inputs = tokenizer(src_text, return_tensors="pt").to(DEVICE)
    current_temp = temp_controller.temp

    actor.eval()
    with torch.no_grad():
        gen_ids = actor.generate(**inputs, forced_bos_token_id=tgt_token_id, do_sample=True,
                                  num_return_sequences=k, max_new_tokens=max_new_tokens,
                                  temperature=current_temp)
    hyps = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)

    rewards = rewarder.get_reward(src_text, hyps)
    avg_reward_raw = float(rewards.mean())

    advantages_arr = (rewards - rewards.mean()) / (rewards.std() + 1e-4)
    advantages = torch.tensor(advantages_arr, dtype=torch.float32).to(DEVICE)

    exp_inputs = {name: v.expand(k, -1) for name, v in inputs.items()}

    with actor.disable_adapter(), torch.no_grad():
        ref_chunks = []
        for ci in range(0, k, 8):
            chunk_in = {name: v[ci:ci + 8] for name, v in exp_inputs.items()}
            ref_chunks.append(get_log_probs(actor, tokenizer, chunk_in["input_ids"],
                                             chunk_in["attention_mask"], gen_ids[ci:ci + 8]))
        ref_log_probs = torch.cat(ref_chunks).detach()

    actor.train()
    optimizer.zero_grad()
    curr_log_probs = get_log_probs(actor, tokenizer, exp_inputs["input_ids"],
                                    exp_inputs["attention_mask"], gen_ids)
    ratio = torch.exp(curr_log_probs - ref_log_probs)
    clipped = torch.clamp(ratio, CLIP_LOW, CLIP_HIGH)
    policy_loss = -torch.min(ratio * advantages, clipped * advantages).mean()
    kl = (torch.exp(curr_log_probs - ref_log_probs) - (curr_log_probs - ref_log_probs) - 1).mean()
    total_loss = policy_loss + kl_beta * kl

    loss_val, kl_val = total_loss.item(), kl.item()
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=1.0)
    optimizer.step()
    torch.cuda.empty_cache()

    return loss_val, avg_reward_raw, kl_val, current_temp


def intermediate_eval(actor, tokenizer, df_eval, lang, n_samples=100, batch_size=8, max_new_tokens=128):
    """Cheap, sparse chrF++ check used only to pick the best checkpoint during training."""
    actor.eval()
    tokenizer.src_lang = "eng_Latn"
    tgt_token_id = tokenizer.convert_tokens_to_ids(lang)
    samples = df_eval.sample(n=min(n_samples, len(df_eval)), random_state=42)
    srcs, refs, hyps = samples["src"].tolist(), samples["ref"].tolist(), []

    for i in range(0, len(srcs), batch_size):
        batch = tokenizer(srcs[i:i + batch_size], return_tensors="pt", padding=True, truncation=True).to(DEVICE)
        with torch.no_grad():
            gen_ids = actor.generate(**batch, forced_bos_token_id=tgt_token_id, do_sample=False,
                                      num_beams=4, max_new_tokens=max_new_tokens)
        hyps.extend(tokenizer.batch_decode(gen_ids, skip_special_tokens=True))

    torch.cuda.empty_cache()
    return chrf_corpus(hyps, refs)


def run_training(actor, tokenizer, rewarder, logger, df_train, df_eval, lang, out_dir,
                  n_epochs=3, lr=5e-6, k=12, max_new_tokens_train=64, max_new_tokens_eval=128,
                  kl_beta=0.001, eval_every=50, eval_n_samples=100,
                  eval_batch_size=8, temp_kwargs=None):
    """
    Trains for n_epochs over df_train, intermediate-eval'ing every eval_every steps and
    saving the single highest-chrF++ checkpoint to {out_dir}/adapters_best/{lang}/.
    Returns (best_chrf, best_step).
    """
    from src.temperature import TemperatureController

    temp_controller = TemperatureController(**(temp_kwargs or {}))
    optimizer = AdamW(filter(lambda p: p.requires_grad, actor.parameters()), lr=lr)
    best_chrf, best_step = -float("inf"), -1
    best_adapter_path = os.path.join(out_dir, "adapters_best", lang)
    global_step = 0

    for epoch in range(n_epochs):
        if n_epochs > 1:
            print(f"\n  Epoch {epoch + 1}/{n_epochs}")
        pbar = tqdm(df_train.itertuples(), total=len(df_train), desc=lang)

        for row in pbar:
            loss_val, reward_raw, kl_val, temp = train_step(
                actor, tokenizer, rewarder, optimizer, row.src, lang, temp_controller,
                k=k, max_new_tokens=max_new_tokens_train, kl_beta=kl_beta,
            )

            chrf_val = None
            if global_step % eval_every == 0 and global_step > 0:
                chrf_val = intermediate_eval(actor, tokenizer, df_eval, lang, eval_n_samples,
                                              eval_batch_size, max_new_tokens_eval)
                if chrf_val > best_chrf:
                    best_chrf, best_step = chrf_val, global_step
                    os.makedirs(best_adapter_path, exist_ok=True)
                    actor.save_pretrained(best_adapter_path)
                    print(f"\n  Step {global_step} (ep{epoch+1}) | chrF: {chrf_val:.4f} | "
                          f"temp: {temp:.2f} | New best -- saved")
                else:
                    print(f"\n  Step {global_step} (ep{epoch+1}) | chrF: {chrf_val:.4f} | "
                          f"temp: {temp:.2f} | best: {best_chrf:.4f} @ step {best_step}")
                actor.train()

            logger.log(lang, global_step, loss_val, reward_raw, kl_val, chrf_val, temp)
            if global_step % 10 == 0:
                pbar.set_postfix({"ep": f"{epoch+1}/{n_epochs}", "r_raw": f"{reward_raw:.4f}",
                                   "kl": f"{kl_val:.4f}", "temp": f"{temp:.2f}", "best": f"{best_chrf:.2f}"})
            global_step += 1
            torch.cuda.empty_cache()

    print(f"\n  Training complete. Total steps: {global_step}")
    print(f"  Best chrF: {best_chrf:.4f} at step {best_step}")
    optimizer.state.clear()
    del optimizer
    torch.cuda.empty_cache()
    return best_chrf, best_step
