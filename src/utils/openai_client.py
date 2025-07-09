from openai import OpenAI
from openai.types import Batch

from config import Credentials

def prompt_chat(
    messages: list[dict],
    model: str = "gpt-4.1-nano-2025-04-14",
    temperature: float = 0.7,
    provider: str = "openai",  # 'openai', 'openrouter', 'fireworks'
) -> str:
    """
    Flexible wrapper to send a chat completion prompt to OpenAI, OpenRouter, or Fireworks.
    Automatically selects base_url and API key based on provider.
    """
    provider = provider.lower()
    if provider == "openrouter":
        base_url = "https://openrouter.ai/api/v1"
        api_key = Credentials.openrouter_api_key
    elif provider == "fireworks":
        base_url = "https://api.fireworks.ai/inference/v1"
        api_key = Credentials.fw_api_key
    elif provider == "openai":
        base_url = "https://api.openai.com/v1"
        api_key = Credentials.openai_api_key
    elif provider == "runpod":
        base_url ='https://api.runpod.ai/v2/j8erq8xjlg68rh/openai/v1'
        api_key = Credentials.runpod_api_key
    else:
        raise ValueError(f"Unknown provider: {provider}")

    client = OpenAI(api_key=api_key, base_url=base_url)
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        extra_body={'reasoning': {'exclude': True}, 'provider': {'sort': 'throughput'}} if provider == 'openrouter' else {},
    )
    return response.choices[0].message.content.strip()

def prompt_chat_structured(
    messages: list[dict],
    text_format,
    model: str = "gpt-4.1-nano-2025-04-14",
    temperature: float = 0.7,
):
    """
    Simple wrapper to send a chat completion prompt and return the response text.
    """
    client = OpenAI(api_key=Credentials.openai_api_key)
    response = client.responses.parse(
        model=model,
        input=messages,
        temperature=temperature,
        text_format=text_format,
    )
    return response.output_parsed

def upload_batch_file(file_name: str):
    client = OpenAI(api_key=Credentials.openai_api_key)
    with open(file_name, "rb") as file:
        batch_file = client.files.create(
            file=file,
            purpose="batch"
        )
    return batch_file

def create_batch_job(file_name: str, endpoint="/v1/chat/completions") -> Batch:
    """
    Creates a batch job using the file at file_name and a specified endpoint.

    :param file_name: Path to the file to be processed in the batch job.
    :param endpoint: The API endpoint to send the batch request to. Defaults to
                         "/v1/chat/completions".
    :return: Metadata of the created batch job, including job ID.
    """
    client = OpenAI(api_key=Credentials.openai_api_key)
    batch_file = upload_batch_file(file_name)
    batch_job = client.batches.create(
        input_file_id=batch_file.id,
        endpoint=endpoint,
        completion_window="24h"
    )
    return batch_job
