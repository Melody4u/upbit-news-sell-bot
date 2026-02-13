from dataclasses import dataclass
from typing import List, Tuple

import requests


NEGATIVE_KEYWORDS = [
    "hack", "해킹", "제재", "금지", "소송", "파산", "상장폐지", "투자주의", "규제", "investigation"
]


@dataclass
class NewsState:
    last_checked: float = 0.0
    negative_score: int = 0
    headlines: List[str] = None

    def __post_init__(self):
        if self.headlines is None:
            self.headlines = []


def fetch_negative_news_score(
    brave_api_key: str,
    query: str,
    country: str,
    lang: str,
    count: int,
) -> Tuple[int, List[str]]:
    if not brave_api_key:
        return 0, []

    url = "https://api.search.brave.com/res/v1/web/search"
    headers = {
        "Accept": "application/json",
        "X-Subscription-Token": brave_api_key,
    }
    params = {
        "q": query,
        "count": count,
        "country": country,
        "search_lang": lang,
        "freshness": "pd",
    }

    resp = requests.get(url, headers=headers, params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json()

    items = data.get("web", {}).get("results", [])
    texts = []
    for item in items:
        title = item.get("title", "") or ""
        desc = item.get("description", "") or ""
        combined = f"{title} {desc}".strip()
        if combined:
            texts.append(combined)

    score = 0
    for text in texts:
        lowered = text.lower()
        if any(k.lower() in lowered for k in NEGATIVE_KEYWORDS):
            score += 1

    return score, texts[:5]
