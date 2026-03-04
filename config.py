import os
from dotenv import load_dotenv
import torch

# Load environment variables from .env file
load_dotenv()

# Groq API configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in environment variables")

# Model configuration
MODEL_NAME = "unsloth/Qwen2.5-VL-7B-Instruct-GGUF"
MODEL_FILE = "Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf"
MMPROJ_FILE = "mmproj-F16.gguf"

# Server configuration
HOST = "0.0.0.0"
PORT = 8000
LOG_LEVEL = "info"

# Generation parameters
# for vlm
MAX_NEW_TOKENS = 300

# for model
TEMPERATURE = 0.3
MAX_COMPLETION_TOKENS = 65536
TOP_P = 1
REASONING_EFFORT = "high"

# Batching configuration
ENABLE_BATCHING = True
BATCH_TIMEOUT = 2.0  # seconds to collect requests before processing
MAX_BATCH_SIZE = 8   # maximum number of images in one batch