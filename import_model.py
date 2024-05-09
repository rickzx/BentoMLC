import json
import logging
import shutil
from dataclasses import asdict

import bentoml
from mlc_llm.chat_module import ChatConfig, _inspect_model_lib_metadata_memory_usage
from mlc_llm.interface import jit
from mlc_llm.support.auto_device import detect_device
from mlc_llm.support.download import download_mlc_weights

MODEL_ID = "HF://mlc-ai/Llama-3-8B-Instruct-q4f16_1-MLC"
BENTO_MODEL_TAG = MODEL_ID.lower().split("/")[-1]

logger = logging.getLogger(__name__)


def import_model(model_id, bento_model_tag):
    logger.info(f"Downloading model weights for {model_id} from Huggingface")
    mlc_dir = download_mlc_weights(model_id)
    cfg_dir = mlc_dir / "mlc-chat-config.json"
    device = detect_device("auto")

    with open(cfg_dir, mode="rt", encoding="utf-8") as file:
        json_object = json.load(file)
        chat_config = ChatConfig._from_json(json_object)

    logger.info("Compiling MLC model library on device")
    model_lib = jit.jit(
        model_path=mlc_dir,
        chat_config=asdict(chat_config),
        device=device,
    )
    _inspect_model_lib_metadata_memory_usage(model_lib, cfg_dir)

    with bentoml.models.create(bento_model_tag) as bento_model_ref:
        shutil.copytree(mlc_dir, bento_model_ref.path, dirs_exist_ok=True)
        shutil.copy(model_lib, bento_model_ref.path)


if __name__ == "__main__":
    import_model(MODEL_ID, BENTO_MODEL_TAG)
