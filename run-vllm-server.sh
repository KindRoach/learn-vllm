#!/bin/bash

# export VLLM_LOGGING_LEVEL=DEBUG

vllm serve /models/Qwen3-4B \
    --disable-log-requests \
    --no-enable-prefix-caching
