from ddgs import DDGS
import requests
import os
import textwrap

SAVE_DIR = "data/CaseLaws"


def internet_case_search(query, max_results=5):
    """Search internet for Indian case laws."""
    results = []
    with DDGS() as ddgs:
        for r in ddgs.text(f"{query} Indian case law", max_results=max_results):
            results.append({
                "title": r.get("title", "Unknown Case"),
                "url": r.get("href", "")
            })
    return results


def summarize_url(url):
    """Summarize legal principle from a URL using Ollama."""
    prompt = (
        "Summarize the legal principle discussed in the following webpage. "
        "Do not add facts not present in the source.\n\n"
        f"{url}"
    )

    res = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "mistral:7b",
            "prompt": prompt,
            "stream": False
        }
    )

    return res.json().get("response", "").strip()


def run_case_law_fallback(issue):
    print("\n🌐 Searching internet for case laws...\n")

    results = internet_case_search(issue)

    if not results:
        print("❌ No results found online.")
        return

    case_laws = []

    for i, r in enumerate(results, 1):
        summary = summarize_url(r["url"])
        case_laws.append({
            "title": r["title"],
            "url": r["url"],
            "summary": summary
        })

        print(f"{i}. {r['title']}")
        print(textwrap.fill(summary, 90))
        print("-" * 90)

    choice = input("\nSave these case laws locally? (y/n): ").strip().lower()
    if choice != "y":
        print("❌ Not saved.")
        return

    save_case_laws(case_laws)


def save_case_laws(case_laws, base_dir=SAVE_DIR):
    os.makedirs(base_dir, exist_ok=True)

    for case in case_laws:
        filename = case["title"][:80].replace(" ", "_") + ".txt"
        path = os.path.join(base_dir, filename)

        with open(path, "w", encoding="utf-8") as f:
            f.write(f"TITLE: {case['title']}\n")
            f.write(f"SOURCE: {case['url']}\n\n")
            f.write(case["summary"])

        print(f"✔ Saved: {filename}")

    print("\n✅ Case laws saved. Re-run the case indexer.")


if __name__ == "__main__":
    issue = input("Enter legal issue to search case laws: ").strip()
    run_case_law_fallback(issue)
