from ddgs import DDGS

query = input("Enter legal issue: ")

with DDGS() as ddgs:
    results = ddgs.text(f"{query} site:indiankanoon.org", max_results=5)

print("\nTop Case Law Links:\n")

for i, r in enumerate(results, 1):
    print(f"{i}. {r['title']}")
    print(f"   {r['href']}\n")
