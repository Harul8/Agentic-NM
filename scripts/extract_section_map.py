"""One-off script: extract old-act → new-act section equivalences from BNS/BNSS/BSA full_text."""
import json, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from config import BARE_CHUNKS_V2

with open(BARE_CHUNKS_V2) as f:
    chunks = json.load(f)

NEW_ACTS = {
    "Bharatiya Nyaya Sanhita, 2023": "BNS",
    "Bharatiya Nagarik Suraksha Sanhita, 2023": "BNSS",
    "Bharatiya Sakshya Adhiniyam, 2023": "BSA",
}

PATTERN = re.compile(
    r"[Ss]imilar to [Ss]ection\s+([\w]+(?:[A-Z])?)\s+(?:from|of)\s+Old\s+(IPC|CrPC|Indian Evidence Act|IEA)",
    re.IGNORECASE,
)

found = {}
for v in chunks.values():
    act = v.get("act_name", "")
    short = NEW_ACTS.get(act)
    if not short:
        continue
    ft = (v.get("full_text") or "").strip()
    sec = str(v.get("section_number", ""))
    title = (v.get("section_title") or "").strip()
    for m in PATTERN.finditer(ft):
        old_sec = m.group(1).strip().upper()
        old_act_raw = m.group(2).strip().upper()
        old_act = "IEA" if "EVIDENCE" in old_act_raw else old_act_raw
        key = (old_act, old_sec)
        if key not in found:
            found[key] = (short, sec, title)

ipc_count = sum(1 for k in found if k[0] == "IPC")
crpc_count = sum(1 for k in found if k[0] == "CRPC")
iea_count = sum(1 for k in found if k[0] == "IEA")
print(f"Total mappings found: IPC={ipc_count}  CrPC={crpc_count}  IEA={iea_count}\n")

KEY_IPC = ["302","304","304B","306","307","308","312","313","354","354A","354B","354C","354D","363","364","366","376","376A","376B","376C","376D","377","379","380","384","385","386","392","393","395","396","397","406","409","415","420","421","425","435","436","440","447","451","452","467","468","471","482","483","486","487","489A","489B","489C","489D","498A","499","500","504","506","509","120B","34","149"]
KEY_CRPC = ["41","57","91","125","154","155","156","160","161","162","163","164","167","173","174","176","177","178","179","180","181","197","204","207","209","210","228","235","239","240","245","265","300","313","315","317","320","321","357","360","361","374","378","379","380","386","389","395","397","401","406","407","408","427","432","436","437","438","439","440","451","457","482"]
KEY_IEA = ["3","9","17","24","25","27","30","32","45","46","47","65","65B","67","68","73","101","102","103","104","106","113A","113B","114","118","119","123","125","126","132","133","134","145","154","155","156","157","165"]

print("Key IPC mappings:")
for s in KEY_IPC:
    k = ("IPC", s)
    if k in found:
        short, ns, t = found[k]
        print(f"  IPC {s:8} -> {short} {ns:6}  {t[:60]}")

print("\nKey CrPC mappings:")
for s in KEY_CRPC:
    k = ("CRPC", s)
    if k in found:
        short, ns, t = found[k]
        print(f"  CrPC {s:8} -> {short} {ns:6}  {t[:60]}")

print("\nKey IEA mappings:")
for s in KEY_IEA:
    k = ("IEA", s)
    if k in found:
        short, ns, t = found[k]
        print(f"  IEA {s:8} -> {short} {ns:6}  {t[:60]}")
