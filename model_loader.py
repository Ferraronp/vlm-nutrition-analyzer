import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig

from config import MODEL_NAME, TORCH_DTYPE, DEVICE_MAP


class ModelLoader:
    _instance = None
    _model = None
    _processor = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def load_model(self):
        if self._model is None or self._processor is None:
            print("Loading model and processor...")
            # Quantization config for 4-bit loading
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=TORCH_DTYPE,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )

            # Load model
            self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                MODEL_NAME,
                torch_dtype=TORCH_DTYPE,
                device_map=DEVICE_MAP,
                quantization_config=quantization_config
            )

            # Load processor
            self._processor = AutoProcessor.from_pretrained(MODEL_NAME)
            print("Model and processor loaded successfully!")

        return self._model, self._processor


# Global instance
model_loader = ModelLoader()