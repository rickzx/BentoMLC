import logging
import os
from typing import AsyncGenerator

import bentoml
from fastapi import FastAPI
from mlc_llm.protocol import error_protocol
from mlc_llm.serve.entrypoints import openai_entrypoints
from mlc_llm.serve.server import ServerContext

MODEL_ID = "HF://mlc-ai/Llama-3-8B-Instruct-q4f16_1-MLC"
BENTO_MODEL_TAG = MODEL_ID.lower().split("/")[-1]

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

app = FastAPI()
app.include_router(openai_entrypoints.app)
app.exception_handler(error_protocol.BadRequestError)(
    error_protocol.bad_request_error_handler
)


class ServerContextManager:
    """A wrapper class to manage ServerContext without modifying the original class."""

    def __init__(self):
        self.context = None

    def start(self):
        self.context = ServerContext()
        self.context.__enter__()  # Manually invoke __enter__
        return self.context

    def end(self):
        if self.context:
            # Manually invoke __exit__ with no exception information
            self.context.__exit__(None, None, None)
            self.context = None


@bentoml.service()
@bentoml.mount_asgi_app(app, path="/")
class MLC:

    bentoml_model_ref = bentoml.models.get(BENTO_MODEL_TAG)

    def __init__(self) -> None:
        from mlc_llm.serve import AsyncMLCEngine

        logger.disabled = False
        logger.info("Loading MLC Engine")
        self.engine = AsyncMLCEngine(
            self.bentoml_model_ref.path,
            model_lib=self._get_model_lib(self.bentoml_model_ref.path),
        )
        self.server_ctx_mgr = ServerContextManager()
        self.server_ctx = self.server_ctx_mgr.start()
        self.server_ctx.add_model(MODEL_ID, self.engine)
        logger.info("MLC Engine loaded successfully.")

    @bentoml.api
    async def generate(
        self, prompt: str = "Explain superconductors like I'm five years old"
    ) -> AsyncGenerator[str, None]:
        async for outputs in await self.engine.chat.completions.create(
            messages=[{"role": "user", "content": prompt}], model=MODEL_ID, stream=True
        ):
            yield outputs.choices[0].delta.content

    def _get_model_lib(self, directory: str) -> str:
        files = [f for f in os.listdir(directory) if f.endswith(".so")]

        if len(files) == 1:
            return os.path.join(
                directory, files[0]
            )  # Return the full path of the .so file
        elif len(files) > 1:
            raise ValueError("More than one .so file found in the directory.")
        else:
            raise FileNotFoundError("No .so file found in the directory.")
