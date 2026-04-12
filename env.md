```bash

uv pip install vllm --torch-backend=auto --extra-index-url https://wheels.vllm.ai/nightly



CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve /data/home/zhangchen/models/Qwen3.5/Qwen3.5-122B-A10B \
    --tensor-parallel-size 4 --port 7999 --host 0.0.0.0 --gpu-memory-utilization 0.90  --max-num-seqs 40 --max-model-len 65536  \
    --served-model-name Qwen3.5-122B-A10B\
    --mm-processor-cache-type shm \
    --mm-encoder-tp-mode data \
    --reasoning-parser qwen3 \
    --enable-prefix-caching \
    --enable-auto-tool-choice --tool-call-parser qwen3_coder 


CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve /data/home/zhangchen/models/Qwen3.5/Qwen3.5-122B-A10B \
    --tensor-parallel-size 1 --port 7999 --host 0.0.0.0 --gpu-memory-utilization 0.90  --max-num-seqs 40 --max-model-len 65536  \
    --enable-expert-parallel --data-parallel-size 4 \
    --served-model-name Qwen3.5-122B-A10B \
    --mm-encoder-tp-mode data \
    --mm-processor-cache-type shm \
    --reasoning-parser qwen3 \
    --enable-prefix-caching \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder 

vllm bench serve \
  --backend openai-chat \
  --base-url http://127.0.0.1:7999 \
  --endpoint /v1/chat/completions \
  --model /data/home/zhangchen/models/Qwen3.5/Qwen3.5-122B-A10B \
  --served-model-name Qwen3.5-122B-A10B \
  --dataset-name random \
  --random-input-len 40480 \
  --random-output-len 10120 \
  --num-prompts 200 \
  --request-rate 20
```

