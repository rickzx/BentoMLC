<div align="center">
    <h1 align="center">Self-host LLMs with MLC-LLM and BentoML</h1>
</div>

This is a BentoML example project, showing you how to serve and deploy open-source Large Language Models using [MLC-LLM](https://llm.mlc.ai/), a high-performance universal deployment solution that allows native deployment of any large language models with native APIs with compiler acceleration.

See [here](https://github.com/bentoml/BentoML?tab=readme-ov-file#%EF%B8%8F-what-you-can-build-with-bentoml) for a full list of BentoML example projects.

💡 This example is served as a basis for advanced code customization, such as custom model, inference logic or MLC-LLM options. For simple LLM hosting with OpenAI compatible endpoint without writing any code, see [OpenLLM](https://github.com/bentoml/OpenLLM).


## Prerequisites

- You have installed Python 3.8+ and `pip`. See the [Python downloads page](https://www.python.org/downloads/) to learn more.
- You have a basic understanding of key concepts in BentoML, such as Services. We recommend you read [Quickstart](https://docs.bentoml.com/en/1.2/get-started/quickstart.html) first.
- If you want to test the Service locally, you need a Nvidia GPU with at least 16G VRAM.
- (Optional) We recommend you create a virtual environment for dependency isolation for this project. See the [Conda documentation](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) or the [Python documentation](https://docs.python.org/3/library/venv.html) for details.

## Install dependencies

```bash
git clone https://github.com/rickzx/BentoMLC.git
pip install -r requirements.txt && pip install -f -U "pydantic>=2.0"
```

## Run the BentoML Service

We have defined a BentoML Service in `service.py`. Run `bentoml serve` in your project directory to start the Service.

```bash
$ bentoml serve .

2024-05-09T04:01:54-0400 [INFO] [cli] Starting production HTTP BentoServer from "service:MLC" listening on http://localhost:3000 (Press CTRL+C to quit)
2024-05-09T04:01:56-0400 [INFO] [entry_service:MLC:1] Loading MLC Engine
[04:02:01] /workspace/mlc-llm/cpp/serve/config.cc:601: Under mode "local", max batch size will be set to 4, max KV cache token capacity will be set to 8192, prefill chunk size will be set to 1024. 
[04:02:01] /workspace/mlc-llm/cpp/serve/config.cc:601: Under mode "interactive", max batch size will be set to 1, max KV cache token capacity will be set to 8192, prefill chunk size will be set to 1024. 
[04:02:01] /workspace/mlc-llm/cpp/serve/config.cc:601: Under mode "server", max batch size will be set to 80, max KV cache token capacity will be set to 125706, prefill chunk size will be set to 1024. 
[04:02:01] /workspace/mlc-llm/cpp/serve/config.cc:678: The actual engine mode is "local". So max batch size is 4, max KV cache token capacity is 8192, prefill chunk size is 1024.
[04:02:01] /workspace/mlc-llm/cpp/serve/config.cc:683: Estimated total single GPU memory usage: 5736.325 MB (Parameters: 4308.133 MB. KVCache: 1092.268 MB. Temporary buffer: 335.925 MB). The actual usage might be slightly larger than the estimated number.
2024-05-09T04:02:04-0400 [INFO] [entry_service:MLC:1] MLC Engine loaded successfully.
```

The server is now active at [http://localhost:3000](http://localhost:3000/). You can interact with it using the Swagger UI or in other different ways.

<details>

<summary>CURL</summary>

```bash
curl -X 'POST' \
  'http://localhost:3000/generate' \
  -H 'accept: text/event-stream' \
  -H 'Content-Type: application/json' \
  -d '{
  "prompt": "Explain superconductors like I'\''m five years old",
  "tokens": null
}'
```

</details>

<details>

<summary>Python client</summary>

```python
import bentoml

with bentoml.SyncHTTPClient("http://localhost:3000") as client:
    response_generator = client.generate(
        prompt="Explain superconductors like I'm five years old",
        tokens=None
    )
    for response in response_generator:
        print(response)
```

</details>

<details>

<summary>OpenAI-compatible endpoints</summary>

See [openai_compatibility.ipynb](https://github.com/rickzx/BentoMLC/blob/main/openai_compatibility.ipynb)

This Service uses the FastAPI [ASGI Integration](https://docs.bentoml.com/en/latest/guides/asgi.html) to hook up MLC-LLM OpenAI-compatible endpoints to the BentoML service.

```python
from openai import OpenAI

client = OpenAI(base_url='http://localhost:3000/v1', api_key='na')

# Use the following func to get the available models
client.models.list()

chat_completion = client.chat.completions.create(
    model="HF://mlc-ai/Llama-3-8B-Instruct-q4f16_1-MLC",
    messages=[
        {
            "role": "user",
            "content": "Explain superconductors like I'm five years old"
        }
    ],
    stream=True,
)

for chunk in chat_completion:
    # Extract and print the content of the model's reply
    print(chunk.choices[0].delta.content or "", end="")
```

**Note**: If your Service is deployed with [protected endpoints on BentoCloud](https://docs.bentoml.com/en/latest/bentocloud/how-tos/manage-access-token.html#access-protected-deployments), you need to set the environment variable `OPENAI_API_KEY` to your BentoCloud API key first.

```bash
export OPENAI_API_KEY={YOUR_BENTOCLOUD_API_TOKEN}
```

You can then use the following line to replace the client in the above code snippet. Refer to [Obtain the endpoint URL](https://docs.bentoml.com/en/latest/bentocloud/how-tos/call-deployment-endpoints.html#obtain-the-endpoint-url) to retrieve the endpoint URL.

```python
client = OpenAI(base_url='your_bentocloud_deployment_endpoint_url/v1')
```

</details>
