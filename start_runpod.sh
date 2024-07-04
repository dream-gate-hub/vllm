/workspace/vllm_env/bin/python3 /workspace/vllm/vllm/entrypoints/openai/api_server.py \
    --model /workspace/Llama-3-Lumimaid-8B-v0.1 \
    --tensor-parallel-size 1 \
    --port 6006 \
    --gpu-memory-utilization 0.9 \
    --chat-template /workspace/Llama-3-Lumimaid-8B-v0.1/template_tipsy_v1.jinja