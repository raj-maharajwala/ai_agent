# sudo apt install python3.12-venv
python3.12 -m venv .venv_agent
source .venv_agent/bin/activate


# Manually download the ollama 
brew install ollama
brew services start ollama

# update ollama
brew update
brew upgrade ollama

# Mac
CMAKE_ARGS="-DLLAMA_METAL=on" FORCE_CMAKE=1 pip install -U llama-cpp-python --no-cache-dir


# Windows
$env:CMAKE_ARGS = "-DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS"
pip install llama-cpp-python


pip install -r raw_requirements.txt
pip freeze > requirements.txt

ollama ps


streamlit run src/app.py



----

ollama -v
ollama version is 0.6.4

pip show ollama
Name: ollama
Version: 0.4.7
Summary: The official Python client for Ollama.