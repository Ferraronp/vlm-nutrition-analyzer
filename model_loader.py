import torch
from huggingface_hub import hf_hub_download
from llama_cpp import Llama
from llama_cpp.llama_chat_format import Qwen25VLChatHandler

from config import MODEL_NAME, MODEL_FILE, MMPROJ_FILE


class ModelLoader:
    _instance = None
    _model = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def load_model(self):
        if self._model is None:
            print("Loading model...")
            model_path = hf_hub_download(repo_id=MODEL_NAME, filename=MODEL_FILE)
            mmproj_path = hf_hub_download(repo_id=MODEL_NAME, filename=MMPROJ_FILE)
            handler = Qwen25VLChatHandler(clip_model_path=mmproj_path)

            # Load model
            self._model = Llama(
                model_path=model_path,
                chat_handler=handler,
                n_ctx=2048,  # Размер контекста
                n_threads=8,  # Количество потоков CPU
                n_gpu_layers=-1,  # -1 для использования всех слоев на GPU (если есть GPU)
                verbose=False,
                logits_all=True
            )

            print("Model loaded successfully!")

        return self._model


# Global instance
model_loader = ModelLoader()