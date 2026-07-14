import json
import glob

for p in glob.glob("../data/sections/*_sections.json"):
    with open(p, encoding="utf-8") as f:
        d = json.load(f)
    print(p, len(d["sections"]))