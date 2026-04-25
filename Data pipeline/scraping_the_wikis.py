from dotenv import load_dotenv
import os
import json
import pandas as pd
from http_client import create_retry_session, DEFAULT_TIMEOUT

pd.options.mode.chained_assignment = None

load_dotenv()

keywords = pd.read_excel("keywords.xlsx")

WIKIPEDIA_API = "https://en.wikipedia.org/w/api.php"


HEADERS = {
    "User-Agent": "PythonAITutor/1.0 (educational project; contact: immawalkrightin@gmail.com)",
    "Accept": "application/json",
    "Accept-Language": "en-US,en;q=0.9",
}

session = create_retry_session(headers=HEADERS)


def search_wikipedia(keyword, num_pages=1):
    """Search Wikipedia for a keyword and return a list of page titles."""
    params = {
        "action": "query",
        "list": "search",
        "srsearch": keyword,
        "srlimit": num_pages,
        "format": "json",
    }
    response = session.get(WIKIPEDIA_API, params=params, timeout=DEFAULT_TIMEOUT)
    response.raise_for_status()
    results = response.json()
    return [item["title"] for item in results["query"]["search"]]


def fetch_wikipedia_page(title):
    """Fetch the full plain-text content and URL of a Wikipedia page by title."""
    params = {
        "action": "query",
        "titles": title,
        "prop": "extracts|info",
        "explaintext": True,
        "inprop": "url",
        "format": "json",
    }
    response = session.get(WIKIPEDIA_API, params=params, timeout=DEFAULT_TIMEOUT)
    response.raise_for_status()
    data = response.json()

    pages = data["query"]["pages"]
    page = next(iter(pages.values()))

    if "missing" in page:
        print(f"  [!] Page not found: {title}")
        return None

    return {
        "url": page.get("fullurl", f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"),
        "title": page.get("title", title),
        "raw_text": page.get("extract", ""),
    }


if not os.path.exists(os.getenv("DATASET_STORAGE_FOLDER")):
    os.makedirs(os.getenv("DATASET_STORAGE_FOLDER"))

output_path = os.getenv("DATASET_STORAGE_FOLDER") + "data.txt"

collected = []
seen_urls = set()  

for ind in keywords.index:
    keyword = keywords.loc[ind, "Keyword"]
    num_pages = int(keywords.loc[ind, "Pages"])

    print(f"Searching Wikipedia for: '{keyword}' ({num_pages} page(s))")

    titles = search_wikipedia(keyword, num_pages)

    for title in titles:
        print(f"  Fetching: {title}")
        page_data = fetch_wikipedia_page(title)

        if not page_data:
            continue

        if page_data['url'] in seen_urls:
            print(f"  [!] Skipped duplicate: {page_data['url']}")
            continue

        seen_urls.add(page_data['url'])
        collected.append(page_data)
        print(f"  Done: {page_data['url']}")

print(f"\nTotal pages scraped: {len(collected)}")
print(f"Writing to: {output_path}")

with open(output_path, "w", encoding="utf-8") as f:
    for item in collected:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print("Done!")
