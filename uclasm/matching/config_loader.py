import yaml

class Config:
    _config = None

    @classmethod
    def load_config(cls, file_path):
        """Load and parse the YAML configuration file."""
        with open(file_path, 'r') as f:
            cls._config =  f.read()

    @classmethod
    def get_config(cls):
        """Return the loaded configuration."""
        if cls._config is None:
            raise ValueError("Configuration has not been loaded yet. Call load_config(file_path) first.")
        return cls._config

    # @classmethod
    # def get(cls, key, default=None):
    #     """Retrieve a specific configuration value by key."""
    #     return cls._config.get(key, default)

# Example usage within the same module (for illustration purposes):
# Config.load_config("path/to/config.yaml")
# config_data = Config.get_config()
# gpu_id = Config.get("gpu_id")
