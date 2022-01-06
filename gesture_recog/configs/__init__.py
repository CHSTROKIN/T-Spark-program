def get_default(key, config, default):
    return config[key] if key in config else default

