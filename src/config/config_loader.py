# src/config/config_loader.py
import os
import yaml
import streamlit_authenticator as stauth
from typing import Dict, List, Optional

class ConfigLoader:
    def __init__(self, config_dir: str = '/mnt/data/ai_chatbot/code.acrivon/AP3-Chatbot/config'):
        self.config_dir = config_dir
        self.configs = {}
        self.authenticator = None

    def load_config(self, yaml_files: List[str]) -> Dict:
        """Load multiple YAML configuration files."""
        for yaml_file in yaml_files:
            config_path = os.path.join(self.config_dir, f'{yaml_file}.yaml')
            try:
                with open(config_path, 'r') as file:
                    self.configs[yaml_file] = yaml.safe_load(file)
            except Exception as e:
                print(f"Error loading config {yaml_file}: {e}")
                self.configs[yaml_file] = {}
        return self.configs

    def get_config(self, config_name: str) -> Dict:
        """Get specific configuration by name."""
        return self.configs.get(config_name, {})
    
    def init_authentication(self) -> Optional[stauth.Authenticate]:
        """Initialize authentication with proper configuration"""
        try:
            config_path = os.path.join(self.config_dir, 'auth.yaml')
            if not os.path.exists(config_path):
                raise FileNotFoundError(f"Auth config not found at {config_path}")
                
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)

            # Initialize authenticator with required parameters
            authenticator = stauth.Authenticate(
                credentials=config_path,
                cookie_name=config['cookie']['name'],
                key=config['cookie']['key'],
                cookie_expiry_days=config['cookie']['expiry_days'],
                auto_hash = True
            )
            return authenticator

        except Exception as e:
            print(f"Authentication initialization error: {e}")
            return None