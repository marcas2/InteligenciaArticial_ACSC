from urllib.parse import urljoin
import requests
from bs4 import BeautifulSoup


BASE_URL = "http://172.16.10.200:5002/"
NORMAL_URL = urljoin(BASE_URL, "Audios/normal/")
ANORMAL_URL = urljoin(BASE_URL, "Audios/anormal/")


def list_wav_urls(folder_url: str) -> list[str]:
    resp = requests.get(folder_url, timeout=30)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")
    urls = []

    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if href.lower().endswith(".wav"):
            urls.append(urljoin(folder_url, href))

    # quitar duplicados conservando orden
    seen = set()
    result = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            result.append(u)

    return result


def download_audio(url: str) -> bytes:
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    return resp.content


def get_remote_training_urls() -> tuple[list[str], list[str]]:
    normal_urls = list_wav_urls(NORMAL_URL)
    anormal_urls = list_wav_urls(ANORMAL_URL)
    return normal_urls, anormal_urls