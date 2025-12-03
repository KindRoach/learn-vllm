# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os

from vllm import LLM, SamplingParams

# enable torch profiler, can also be set on cmd line
os.environ["VLLM_TORCH_PROFILER_DIR"] = "./vllm_profile"

# Sample prompts.
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]

# Create a sampling params object.
sampling_params = SamplingParams(
    min_tokens=256,
    max_tokens=256,
)

def main():
    # Create an LLM.
    model_name = "/models/Qwen3-4B"
    llm = LLM(
        model=model_name,
        distributed_executor_backend="mp"
    )

    llm.start_profile()

    _ = llm.generate(prompts, sampling_params)

    llm.stop_profile()


if __name__ == "__main__":
    main()
