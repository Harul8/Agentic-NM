import re


def detect_judgment_template(first_page_text: str) -> str:
    """
    Detect the format/template of an Indian judgment
    based on the first page text (STRICTLY first page only).
    """

    if not first_page_text:
        return "UNKNOWN_TEMPLATE"

    text = first_page_text.upper()

    # ------------------------------------------------
    # Supreme Court Registry - Record of Proceedings
    # ------------------------------------------------
    if "RECORD OF PROCEEDINGS" in text:
        return "SC_RECORD_OF_PROCEEDINGS"

    # ------------------------------------------------
    # Supreme Court final judgments
    # ------------------------------------------------
    if "IN THE SUPREME COURT OF INDIA" in text:

        if "CONSTITUTION BENCH" in text:
            return "SC_CONSTITUTION_BENCH"

        if "CIVIL APPELLATE JURISDICTION" in text:
            return "SC_CIVIL_APPEAL"

        if "CRIMINAL APPELLATE JURISDICTION" in text:
            return "SC_CRIMINAL_APPEAL"

        return "SC_GENERAL_JUDGMENT"

    # ------------------------------------------------
    # Supreme Court Reports (SCR)
    # ------------------------------------------------
    if re.search(r"\d{4}\s+SCR\s+\d+", text):
        return "SCR_REPORT"

    # ------------------------------------------------
    # SCC Reporter
    # ------------------------------------------------
    if re.search(r"\(\d{4}\)\s+\d+\s+SCC\s+\d+", text):
        return "SCC_REPORT"

    # ------------------------------------------------
    # AIR Reporter (Supreme Court)
    # ------------------------------------------------
    if re.search(r"A\.?I\.?R\.?\s+\d{4}\s+SC\s+\d+", text):
        return "AIR_REPORT"

    # ------------------------------------------------
    # High Court judgments
    # ------------------------------------------------
    if "IN THE HIGH COURT" in text:

        if "WRIT PETITION" in text:
            return "HC_WRIT"

        if "CRIMINAL PETITION" in text:
            return "HC_CRIMINAL"

        if "WRIT APPEAL" in text:
            return "HC_WRIT_APPEAL"

        return "HC_GENERAL"

    # ------------------------------------------------
    # Telangana High Court specific pattern
    # ------------------------------------------------
    if "HIGH COURT FOR THE STATE OF TELANGANA" in text:
        return "TELANGANA_HC"

    # ------------------------------------------------
    # Fallback
    # ------------------------------------------------
    return "UNKNOWN_TEMPLATE"

