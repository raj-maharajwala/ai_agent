from typing import List, Dict
import os

TASK_OPTIONS: List[str] = [
    "Auto Selection",
    "Smart Search Agent",
    "Productivity Assistant",
    "Data Analysis Agent"
]

DEFAULT_MODEL_PARAMS: Dict[str, any] = {
    "temperature": 0.1, #0.0,
    "top_k": 20, #15,
    "top_p": 0.4, #0.5,
    "repeat_penalty": 1.3,
    "n_ctx": 8192,
    "n_batch": 512,
    "n_gpu_layers": -1,
    "n_threads":os.cpu_count()
}