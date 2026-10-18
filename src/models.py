"""
Model loading: 4-bit NF4 quantized NLLB-200 base + rank-16 LoRA adapter on q_proj/v_proj.
"""
import torch
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForSeq2SeqLM, BitsAndBytesConfig

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def make_bnb_config():
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )


def make_lora_config(r=16, alpha=32, dropout=0.05, target_modules=("q_proj", "v_proj")):
    return LoraConfig(
        r=r, lora_alpha=alpha, target_modules=list(target_modules),
        lora_dropout=dropout, task_type="SEQ_2_SEQ_LM",
    )


def load_base_model(model_name):
    return AutoModelForSeq2SeqLM.from_pretrained(
        model_name, quantization_config=make_bnb_config(),
        torch_dtype=torch.bfloat16, device_map={"": DEVICE},
    )


def load_actor(model_name, lora_r=16, lora_alpha=32, lora_dropout=0.05, init_adapter_path=None):
    """
    Loads the trainable policy: 4-bit base + LoRA adapter. If init_adapter_path is given
    (SFT->GRPO cascade), continues GRPO from that existing adapter instead of a fresh one.
    """
    base = prepare_model_for_kbit_training(load_base_model(model_name), use_gradient_checkpointing=False)
    if init_adapter_path:
        print(f"  Initializing GRPO LoRA from existing adapter: {init_adapter_path}")
        actor = PeftModel.from_pretrained(base, init_adapter_path, is_trainable=True)
    else:
        actor = get_peft_model(base, make_lora_config(lora_r, lora_alpha, lora_dropout))
    actor.enable_input_require_grads()
    return actor


def load_vanilla_model(model_name):
    """Frozen, un-adapted base model for scoring/comparison."""
    model = load_base_model(model_name)
    model.eval()
    return model
