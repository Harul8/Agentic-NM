import pandas as pd

from sync_feedback_log import EXCEL_PATH, EXCEL_HEADERS_26, excel_to_html


def main() -> None:
    notes_row = [
        "e.g. Mar 1, 2026 10:15 AM",
        "",
        "Raw user query",
        "Disputes from model",
        "Act § No",
        "Bullet list of info requested",
        "Case citations",
        "Full legal opinion",
        "Router classification (model)",
        "AI's disputes",
        "AI's sections",
        "AI's additional info",
        "AI's case laws",
        "AI's legal opinion",
        "AI's router classification",
        "Your disputes",
        "Your sections",
        "Your additional info",
        "Your case laws",
        "Your legal opinion",
        "Your router classification",
        "Enumerated issues",
        "Corrective guidance",
        "New rule derived",
        "Yes/No/In Progress",
        "1–5",
    ]

    # First row blank, second row headers, third row notes; no data rows yet.
    rows = [
        [""] * 26,
        EXCEL_HEADERS_26,
        notes_row,
    ]

    df = pd.DataFrame(rows)
    df = df.iloc[:, :26]
    df.to_excel(EXCEL_PATH, index=False, header=False)

    # Regenerate HTML from the fresh Excel (header/notes only).
    excel_to_html()

    print("Feedback log reset: Excel and HTML now contain only header/notes rows.")


if __name__ == "__main__":
    main()

