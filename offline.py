import os
import random

from vllm import SamplingParams, LLM
from transformers import AutoTokenizer


def generate_random_inputs(model_id_or_path: str, token_num: int) -> list[int]:
    """
    Generate random token ids by given token number.

    Args:
        model_id_or_path: HuggingFace model ID or path to get the tokenizer
        token_num: Number of tokens to generate

    Returns:
        Generated token ids as a list[int]
    """
    if not isinstance(token_num, int) or token_num <= 0:
        raise ValueError(f"token_num must be a positive int, got: {token_num!r}")

    # Load tokenizer from HuggingFace
    tokenizer = AutoTokenizer.from_pretrained(model_id_or_path)

    # Sample from actual vocab ids (HF vocab ids are not guaranteed to be contiguous).
    vocab = tokenizer.get_vocab()  # token -> id
    vocab_ids = list(vocab.values())
    if not vocab_ids:
        raise ValueError("Tokenizer vocab is empty; cannot generate random inputs.")

    # Avoid sampling special tokens (e.g., EOS/PAD), since they can affect
    # generation behavior in unintuitive ways.
    special_ids = set(getattr(tokenizer, "all_special_ids", []) or [])
    candidate_ids = [tid for tid in vocab_ids if tid not in special_ids]
    if not candidate_ids:
        candidate_ids = vocab_ids

    # random.choices is faster than repeated random.choice for large k.
    return random.choices(candidate_ids, k=token_num)


def main():
    model_name = "/models/Qwen3-4B"

    num_req = 16
    num_input_tokens = 16 * 1024

    prompts = [
        {
            "prompt_token_ids": generate_random_inputs(model_name, num_input_tokens),
        }
        for _ in range(num_req)
    ]

    sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

    enable_profile = True
    if enable_profile:
        # enable torch profiler, can also be set on cmd line
        os.environ["VLLM_TORCH_PROFILER_DIR"] = "./vllm_profile"

    llm = LLM(
        model=model_name,
        # distributed_executor_backend="mp",
        # enforce_eager=True
    )

    if enable_profile:
        # warmup
        outputs = llm.generate(prompts, sampling_params)

        # start profiling
        llm.start_profile()

    # actual generation
    outputs = llm.generate(prompts, sampling_params)

    if enable_profile:
        # stop profiling
        llm.stop_profile()


if __name__ == "__main__":
    main()
