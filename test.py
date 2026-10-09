import requests, json, pandas as pd

ROOT = "https://exam.sanand.workers.dev/questionData?email=24f3003188%40ds.study.iitm.ac.in&quizSign=0eyEF2vP3oDh3px4Vo9tHq71IUREcIBMw1EaaKOnFW%2B1UjL9P9wdpb%2BeSpssg%2BTBBUrwhwiNQq6eP7rfINIEW02t%2BV2%2B6VRmhVnmIR1bUNbnmebNkceN9GqbctO9qMtZM34SaH835zl934rGiZU2buVnN4k0ukAhSOgidNU937swfeIxT4C7DGTXSlKOixvebm%2BogWfa428bTAnEjTWD0Anq74Swn8jsnnq3qaNQm4tKka%2BfUxh8M3iq5kS4h%2FX1A3ONm1qNCal6lJxKAP%2FuTWM94UXVF8n9GHUdQ72GmNL5jBEudtFnRyFiECqe5%2B%2FopPut9MsHAywOHqoxDwCwgQ%3D%3D&questionId=q-ledger-agent-server&path=/"
links = requests.get(ROOT).json()["links"]

# export
exp = [json.loads(l) for l in requests.get(links["export"]).text.splitlines() if l.strip()]
e = pd.DataFrame(exp)

# orders: follow "next" until it is missing
rows, url = [], links["orders"]
while url:
    p = requests.get(url).json()
    rows += p["orders"]
    url = p.get("next")
o = pd.DataFrame(rows)

print("export rows:", len(e), "unique ids:", e.id.nunique())
print("orders rows:", len(o), "unique ids:", o.id.nunique())
print("same ids?", set(e.id) == set(o.id))

for col in ["status", "currency", "region", "product"]:
    print(col, sorted(e[col].astype(str).unique()))
print("nulls:\n", e.isna().sum())
print("ids with stray spaces/lowercase:", (e.id != e.id.str.strip().str.upper()).sum())