"""
Build the rep app: puts outputs/app_data.json into app/template.html -> app/tapline.html.
Run after step 10:   python app/build_app.py
The result is one self-contained HTML file (no server needed).
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

with open(os.path.join(ROOT, "outputs", "app_data.json"), encoding="utf-8") as fh:
    data = json.load(fh)
payload = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")   # safe inside <script>

with open(os.path.join(HERE, "template.html"), encoding="utf-8") as fh:
    html = fh.read().replace("__DATA__", payload)

out = os.path.join(HERE, "tapline.html")
with open(out, "w", encoding="utf-8") as fh:
    fh.write(html)
print(f"wrote {out} ({os.path.getsize(out) / 1024:.0f} KB, {len(data['accounts'])} accounts)")
