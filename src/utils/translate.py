import requests
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type, before_sleep_log
import logging

from config import Credentials

logger = logging.getLogger(__name__)

class RateLimitError(requests.exceptions.HTTPError):
    """Custom to detect HTTP 429 separately if needed."""


def raise_for_429(response):
    if response.status_code == 429:
        raise RateLimitError(f"429 Too Many Requests: {response.text}", response=response)
    response.raise_for_status()

@retry(
    retry=(
        retry_if_exception_type(requests.exceptions.JSONDecodeError) |
        retry_if_exception_type(RateLimitError)
    ),
    wait=wait_exponential(multiplier=2, min=2, max=30),
    stop=stop_after_attempt(10),
    before_sleep=before_sleep_log(logger, logging.WARNING)
)
def translate(text: str, source_lang: str = 'EN', target_lang: str = 'DE') -> str:
    response = requests.post(
        "https://api-free.deepl.com/v2/translate",
        data={
            "auth_key": Credentials.deepl_api_key,
            "text": text,
            "source_lang": source_lang,
            "target_lang": target_lang,
            "tag_handling": "html",
            "outline_detection": "0"
        }
    )
    raise_for_429(response)
    result = response.json()
    return result['translations'][0]['text']