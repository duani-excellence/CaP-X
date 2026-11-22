"""Small public choice catalog; provider credentials stay on the server."""
MODELS = [
    {'provider': 'deepseek', 'model': 'deepseek-v4-pro', 'label': 'DeepSeek V4 Pro'},
    {'provider': 'openai', 'model': 'gpt-6-astra', 'label': 'GPT-6 Astra'},
]
OBSERVATIONS = [
    {'mode': 'vision', 'label': '视觉观测', 'config_path': 'env_configs/cube_stack/franka_robosuite_cube_stack_web.yaml'},
    {'mode': 'privileged', 'label': '仿真真值', 'config_path': 'env_configs/cube_stack/franka_robosuite_cube_stack_privileged_web.yaml'},
]


def resolve_selection(provider, mode):
    from capx.llm.providers import provider_config
    choice = next((x for x in MODELS if x['provider'] == provider), None)
    observation = next((x for x in OBSERVATIONS if x['mode'] == mode), None)
    if choice is None or observation is None:
        raise ValueError('请选择模型和观测模式')
    url, model, key = provider_config(provider)
    if model != choice['model']:
        raise ValueError(f'{provider} 配置的模型与所选模型不一致，请检查服务器配置')
    return url, model, key, observation['config_path']
