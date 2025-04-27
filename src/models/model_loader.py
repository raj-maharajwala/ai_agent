from llama_cpp import Llama
# from ollama_functions_custom import OllamaFunctions
from typing import Optional, Dict
from utils.paths import PathManager # CUSTOM_CLASS
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_openai import ChatOpenAI
# from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
from langchain_ollama.llms import OllamaLLM
import torch
import os

class ModelLoader:
    def __init__(self, configs: Dict):
        self.configs = configs
        self.path_manager = PathManager()
        self.model_paths = self.path_manager.model_paths
    def load_model(self, task: str) -> any:
        """Load appropriate model based on task and parameters."""
        
        if task == 'Productivity Assistant':
            return self._load_reasoning(
                model_path = str(self.model_paths["OLLAMA_Deepseek-R1"]),
                base_url = 'http://127.0.0.1:11434'
                # **model_params
            )
        
        elif task == 'Auto Selection':
            self._load_default_ollama(
            # base_url=self.configs['model']['end_point_local'],
            model_path = str(self.model_paths["OLLAMA_QWEN"]),
            base_url = 'http://127.0.0.1:11434'
            )
            
        return self._load_default_ollama(
            # base_url=self.configs['model']['end_point_local'],
            model_path = str(self.model_paths["OLLAMA_QWEN"]),
            base_url = 'http://127.0.0.1:11434'
            # **model_params
        )
    
    def _load_reasoning(self, model_path: str, base_url: str) -> OllamaLLM:  #  **kwargs
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return OllamaLLM(
                base_url=base_url,
                model=model_path,
                top_k=20,
                top_p=0.4,
                temperature=0.0,
                num_ctx = 29184,
                verbose=True,
                callbacks=[StreamingStdOutCallbackHandler()],
                streaming=True,
                seed = -1,
                num_gpu=-2, # Explicitly use one device GPU (Total - 1)
                keep_alive=-1,
                f16=True,
                # mirostat=1,  # Added adaptive sampling for better quality
                # mirostat_tau=4.0,  # Conservative tau value for factual responses
                # mirostat_eta=0.1  # Learning rate for mirostat
            )

    def _load_default_ollama(self, model_path: str, base_url: str) -> ChatOllama:  # , **kwargs
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"
        return ChatOllama(
            base_url=base_url,
            model=model_path,
            top_k=30,  # Increased from 20 to consider more token candidates
            top_p=0.4,  
            temperature=0.03,  # Reduced slightly for more deterministic/factual responses
            num_ctx=15000,#19184,  # Increased context window to maximum supported
            verbose=False,
            format='json',
            num_gpu=-2,  # Explicitly use one device GPU (Total - 1)
            keep_alive=-1,
            seed=42,  # Fixed seed for reproducibility
            f16=True,  
            mirostat=1,  # Added adaptive sampling for better quality
            mirostat_tau=4.0,  # Conservative tau value for factual responses
            mirostat_eta=0.1  # Learning rate for mirostat
        )

###############
"""
sudo systemctl stop ollama
sudo nano /etc/systemd/system/ollama.service
[Service]
Environment="OLLAMA_FLASH_ATTENTION=1"
Environment="OLLAMA_KV_CACHE_TYPE=q8_0"
Environment="OLLAMA_MAX_LOADED_MODELS=3"
Environment="OLLAMA_NUM_PARALLEL=4"
Environment="OLLAMA_NUMA=1"
# Environment="OLLAMA_KEEP_ALIVE=-1"
sudo systemctl daemon-reload
sudo systemctl restart ollama
---
torch.cuda.empty_cache()
---
sudo journalctl -u ollama.service | grep "kv_cache"
---
sudo systemctl stop ollama
---
OLLAMA_KEEP_ALIVE=-1 OLLAMA_FLASH_ATTENTION=true OLLAMA_KV_CACHE_TYPE=q8_0 OLLAMA_MAX_LOADED_MODELS=3 ollama serve
(or)
export OLLAMA_MAX_LOADED_MODELS=3
export OLLAMA_NUM_PARALLEL=4
export OLLAMA_FLASH_ATTENTION=true
export OLLAMA_KV_CACHE_TYPE=q8_0
ollama serve
---
nvidia-smi -l 1
---
"""
#####