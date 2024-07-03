/root/miniconda3/envs/vllm/bin/python3 -m vllm.entrypoints.openai.api_server \
    --model /root/autodl-tmp/Llama-3-Lumimaid-8B-v0.1 \
    --tensor-parallel-size 1 \
    --port 6006 \
    --gpu-memory-utilization 0.9 \
    --chat-template /root/autodl-tmp/Llama-3-Lumimaid-8B-v0.1/template_tipsy_v1.jinja