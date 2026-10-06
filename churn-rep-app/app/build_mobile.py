"""
Build the mobile rep app: packs outputs/mobile_data.json into app/mobile_template.html -> app/fieldline.html.
Run after step 11:   python app/build_mobile.py

Every active account is included, so the data is packed to keep the page light on a phone:
columns are listed once, and repeated texts (segments, reasons, actions, offers, cities,
product lines) are stored once in a lookup table and referred to by number.
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

COLS = ["id", "city", "state", "lat", "lng", "rank", "segment", "risk", "annual_value", "value_at_risk",
        "reasons", "n_orders", "days_since_last", "usual_gap_days", "orders_last_90d", "usual_orders_per_90d",
        "late_share", "last_order_late", "freight_share", "avg_review", "top_lines", "dropped_line",
        "monthly_spend", "complaint", "action", "offer", "channel"]
TEXT = {"city", "state", "segment", "dropped_line", "action", "offer", "channel"}
ROUND = {"lat": 5, "lng": 5, "risk": 4, "annual_value": 0, "value_at_risk": 0, "usual_gap_days": 1,
         "usual_orders_per_90d": 2, "late_share": 3, "freight_share": 3, "avg_review": 2}

with open(os.path.join(ROOT, "outputs", "mobile_data.json"), encoding="utf-8") as fh:
    data = json.load(fh)

words, index = [], {}
def w(s):
    if s is None:
        return None
    if s not in index:
        index[s] = len(words); words.append(s)
    return index[s]

rows = []
for a in data["accounts"]:
    row = []
    for c in COLS:
        v = a.get(c)
        if c in TEXT:
            v = w(v)
        elif c == "reasons":
            v = [[w(r["text"]), r["points"]] for r in v]
        elif c == "top_lines":
            v = [w(x) for x in v]
        elif c in ROUND and v is not None:
            v = round(v, ROUND[c]) if ROUND[c] else round(v)
        row.append(v)
    rows.append(row)

packed = {"meta": data["meta"], "depot": data["depot"], "route": data["route"],
          "cols": COLS, "text": sorted(TEXT | {"reasons", "top_lines"}), "words": words, "rows": rows}
payload = json.dumps(packed, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")

with open(os.path.join(HERE, "mobile_template.html"), encoding="utf-8") as fh:
    html = fh.read().replace("__DATA__", payload)
out = os.path.join(HERE, "fieldline.html")
with open(out, "w", encoding="utf-8") as fh:
    fh.write(html)
print(f"wrote {out} ({os.path.getsize(out) / 1e6:.1f} MB, {len(rows):,} accounts, {len(data['route']['stops'])} stops)")
