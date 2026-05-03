import json
import urllib.request


class EPOClient:
    """
    Fetch CPC hierarchy from EPO Linked Open Data API.
    """

    def __init__(self, base_url: str = "https://data.epo.org/linked-data/def/cpc"):
        self.base_url = base_url.rstrip("/")

    def fetch_class(self, symbol: str) -> dict:
        url = f"{self.base_url}/{symbol}.json"

        try:
            with urllib.request.urlopen(url, timeout=15) as r:
                return json.loads(r.read().decode("utf-8"))
        except Exception:
            return {}

    def extract_title(self, data: dict) -> str:
        try:
            topic = data["result"]["primaryTopic"]
            title = topic.get("title") or topic.get("fullTitle")

            if isinstance(title, list):
                return title[0]

            return title or ""
        except Exception:
            return ""