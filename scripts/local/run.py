"""Small local entry point. API keys stay in environment, not argv."""
import argparse
import os
from datetime import datetime

from capx.llm.providers import provider_config

def run():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["oracle", "vision-oracle", "privileged", "vision", "web"])
    p.add_argument("--backend", choices=["ollama", "vllm", "openrouter", "cloud", "deepseek", "openai", "chatgpt"], default=os.getenv("CAPX_LLM_BACKEND", "ollama"))
    p.add_argument("--trials", type=int, default=1)
    p.add_argument("--output-dir")
    p.add_argument('--snapshot-manifest')
    p.add_argument('--pilot-phase', choices=['prepare','oracle','models'])
    a = p.parse_args()
    if a.snapshot_manifest:
        if not a.pilot_phase or not a.output_dir:
            p.error('--snapshot-manifest requires --pilot-phase and --output-dir')
        from capx.experiments.pilot import run as pilot_run
        pilot_run(a.snapshot_manifest, a.pilot_phase, a.output_dir)
        return
    configs = {
        "oracle": "env_configs/human_oracle_code/robosuite/franka_robosuite_cube_stack_privileged_oracle.yaml",
        "privileged": "env_configs/cube_stack/franka_robosuite_cube_stack_privileged.yaml",
        "vision-oracle": "env_configs/cube_stack/franka_robosuite_cube_stack.yaml",
        "vision": "env_configs/cube_stack/franka_robosuite_cube_stack.yaml",
        "web": "env_configs/cube_stack/franka_robosuite_cube_stack.yaml",
    }
    endpoints = {
        "ollama": ("http://127.0.0.1:11434/v1/chat/completions", "qwen2.5-coder:7b", None),
        "vllm": ("http://127.0.0.1:8000/v1/chat/completions", "qwen2.5-coder:7b", None),
        "openrouter": ("http://127.0.0.1:8110/chat/completions", os.getenv("CAPX_OPENROUTER_MODEL", ""), None),
        "cloud": (os.getenv("CAPX_CLOUD_SERVER_URL", ""), os.getenv("CAPX_CLOUD_MODEL", ""), (os.getenv("CAPX_CLOUD_API_KEY") or os.getenv("DEEPSEEK_API_KEY", ""))),
    }
    if a.mode in ('oracle', 'vision-oracle'):
        url, model, key = endpoints['ollama']  # unused: no LLM call
    elif a.backend in ('deepseek', 'openai', 'chatgpt'):
        try:
            url, model, key = provider_config(a.backend)
        except ValueError as exc:
            p.error(str(exc))
    else:
        url, model, key = endpoints[a.backend]
    if a.mode not in ("oracle", "vision-oracle"):
        if not url or not model or (a.backend == "cloud" and not key):
            p.error("云端配置缺失：请填写 .env.local 中对应的服务 URL、模型名和 API key。")
    from capx.envs.launch import LaunchArgs, main
    args = LaunchArgs(
        config_path=configs[a.mode], server_url=url, model=model,
        api_key=key, temperature=0.2,
        max_tokens=int(os.getenv("CAPX_CLOUD_MAX_TOKENS", "16384")) if a.backend in ("cloud", "deepseek", "openai", "chatgpt") else 4096,
        total_trials=a.trials, num_workers=1, record_video=True,
        use_oracle_code=a.mode in ("oracle", "vision-oracle"),
        use_visual_feedback=False, use_img_differencing=False,
        use_video_differencing=False, use_parallel_ensemble=False,
        web_ui=a.mode == "web", web_ui_port=8200,
        output_dir=a.output_dir or f"outputs/reproduction/{a.mode}-{a.backend}-{datetime.now():%Y%m%d-%H%M%S}",
    )
    main(args)


if __name__ == "__main__":
    run()
