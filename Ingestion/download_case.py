import requests
from bs4 import BeautifulSoup
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CASELAW_DIR

def download_case(url):
    print("Downloading:", url)
    r = requests.get(url, timeout=15)
    soup = BeautifulSoup(r.text, "html.parser")

    title = soup.title.text.strip()
    filename = re.sub(r"[^a-zA-Z0-9]", "_", title)[:80] + ".txt"

    os.makedirs(CASELAW_DIR, exist_ok=True)
    path = os.path.join(CASELAW_DIR, filename)

    with open(path, "w", encoding="utf-8") as f:
        f.write(title + "\n\n" + text)

    print("Saved to:", path)


if __name__ == "__main__":
    url = input("Paste case law URL: ")
    download_case(url)
