# utils/paths.py
import os
from pathlib import Path
from typing import Dict

class PathManager:
    def __init__(self):
        # Base paths
        self.APP_DIR = Path("/Users/rajmaharajwala/raj-maharajwala-github/ai_agent")
        self.ASSETS_DIR = self.APP_DIR / "assets"
        self.MODELS_DIR = self.APP_DIR / "models"
        self.VECTORDB_DIR = self.APP_DIR / "index_vectordb" / "productivity"

    @property
    def model_paths(self) -> Dict[str, Path]:
        """Returns a dictionary of model paths"""
        return {
            "EMBEDDINGS": self.MODELS_DIR / "snowflake-arctic-embed-l-v2.0",
            "OLLAMA": "qwq:32b-q8_0"
        }

    @property
    def vectordb(self) -> Dict[str, Path]:
        """Returns a dictionary of vectordb paths"""
        return {
            "CSV_PATH": self.VECTORDB_DIR / "sql_examples_productivity.csv",
            "INDEX_PATH": self.VECTORDB_DIR / "sql-example-faiss-productivity-index-L2.index",
            "METADATA_DIR": self.VECTORDB_DIR / "sql-example-productivity-metadata.pkl"
        }

    @property
    def asset_paths(self) -> Dict[str, Path]:
        """Returns a dictionary of asset paths"""
        return {
            "LOGO": self.ASSETS_DIR / "logo.png",
            "CSS": self.ASSETS_DIR / "styles.css",
            # "CONFIG": self.ASSETS_DIR / "config.yaml"
        }