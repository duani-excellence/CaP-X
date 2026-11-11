"""Explicit official providers; credentials never cross provider boundaries."""
import os


def provider_config(name, environ=None):
    env = os.environ if environ is None else environ
    name = 'openai' if name == 'chatgpt' else name
    defaults = {
        'deepseek': ('https://api.deepseek.com/v1/chat/completions', 'deepseek-v4-pro'),
        'openai': ('https://api.openai.com/v1/chat/completions', ''),
    }
    url, model = defaults[name]
    prefix = name.upper()
    url = env.get(prefix + '_SERVER_URL', '').strip() or url
    model = env.get(prefix + '_MODEL', '').strip() or model
    key = env.get(prefix + '_API_KEY', '').strip()
    missing = [field for field, value in [('MODEL', model), ('API_KEY', key)] if not value]
    if missing:
        raise ValueError('请填写 .env.api.local：' + ', '.join(prefix + '_' + field for field in missing))
    return url, model, key
