import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from typing import Optional

DEFAULT_TIMEOUT = 20


def create_retry_session(headers: Optional[dict] = None) -> requests.Session:
    """Create a shared HTTP session with retries for transient failures."""
    session = requests.Session()
    retry = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503],
        allowed_methods=["HEAD", "GET", "OPTIONS"],
    )
    session.mount("https://", HTTPAdapter(max_retries=retry))
    session.mount("http://", HTTPAdapter(max_retries=retry))

    if headers:
        session.headers.update(headers)

    return session
