import json
with open('legal_database/vector_store/bareacts_v2_chunks.json', encoding='utf-8') as f:
    chunks = json.load(f)
bns = [(k,v) for k,v in chunks.items() if 'bharatiya nyaya sanhita' in (v.get('act_name') or '').lower()]
for k,v in sorted(bns, key=lambda x: str(x[1].get('section_number',''))):
    sn = (v.get('section_number') or '').strip()
    try:
        n = int(sn)
        if 83 <= n <= 87:
            title = v.get('section_title','')[:80]
            text = (v.get('full_text') or '')[:120]
            print('Sec ' + sn + ': ' + title)
            print('  ' + text)
    except:
        pass