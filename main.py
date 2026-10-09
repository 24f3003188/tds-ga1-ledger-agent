import os, json, requests, pandas as pd, numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI

ROOT = "PASTE_ROOT_URL"     # the URL ending in path=/
TZ = "Asia/Kolkata"

app = FastAPI()
client = OpenAI(api_key=os.environ["AIPIPE_TOKEN"],
                base_url="https://aipipe.org/openai/v1")

class Q(BaseModel):
    question: str

def load():
    links = requests.get(ROOT, timeout=20).json()["links"]
    rates = requests.get(links["rates"], timeout=20).json()["usd_per_unit"]
    text = requests.get(links["export"], timeout=30).text
    df = pd.DataFrame([json.loads(l) for l in text.splitlines() if l.strip()])

    # 1. keep only the newest version of each order id
    df["updated_at"] = pd.to_datetime(df["updated_at"], utc=True, format="ISO8601")
    df = df.sort_values("updated_at").drop_duplicates("id", keep="last")

    # 2. tidy status text
    df["status"] = df["status"].astype(str).str.strip().str.lower()

    # 3. business dates in Kolkata time
    ts = pd.to_datetime(df["created_at"], utc=True, format="ISO8601").dt.tz_convert(TZ)
    df["month"] = ts.dt.strftime("%Y-%m")        # e.g. 2026-04
    df["date"] = ts.dt.strftime("%Y-%m-%d")

    # 4. money in USD: multiply by "USD per unit"
    df["usd"] = df["amount"].astype(float) * df["currency"].map(rates).astype(float)

    df = df.drop(columns=["created_at", "updated_at"])
    return df.reset_index(drop=True), rates

DF, RATES = load()      # once, at startup

PROMPT = """You write pandas code. A DataFrame `df` already exists.
Columns: {cols}
Sample rows: {sample}
Statuses in the data: {statuses}

Rules:
- `usd` is the order amount already converted to USD. Use it for ALL money. Never convert again.
- `month` is like '2026-04' and `date` like '2026-04-15' (already in the business timezone). Filter dates with these columns only.
- Revenue = sum of `usd` where status == 'paid'. Units sold = sum of `qty` where status == 'paid'.
- Refunds = rows where status == 'refunded'. Refund amount = sum of `usd` for those rows. 'void' rows count for nothing.
- "Top-selling by revenue" = groupby('product')['usd'].sum() then idxmax(). "By units" = use `qty` instead.
- Customers: count with df['customer'].nunique() on the right filtered rows.
- Region values and product names must be matched exactly as in the data; return product names exactly as written.
- Put the final answer in a variable named `result` (a plain number or string). Do not round, do not print.
- Output ONLY raw Python code, no markdown.

Examples:
Q: Total revenue in USD from the North region in March 2026?
result = df[(df.status=='paid') & (df.region=='North') & (df.month=='2026-03')].usd.sum()
Q: Which product had the most revenue in April 2026?
result = df[(df.status=='paid') & (df.month=='2026-04')].groupby('product').usd.sum().idxmax()

Question: {q}"""

def to_json_safe(x):
    if isinstance(x, pd.DataFrame):
        x = x.iloc[0, 0]
    if isinstance(x, (pd.Series, pd.Index)):
        x = x.iloc[0] if len(x) == 1 else x.tolist()
    if isinstance(x, np.generic):
        x = x.item()
    if isinstance(x, float):
        x = round(x, 2)
    return x

def run_code(code):
    code = code.replace("```python", "").replace("```", "").strip()
    ns = {"df": DF.copy(), "rates": RATES, "pd": pd, "np": np}
    exec(code, ns)                         # ONE dictionary
    return to_json_safe(ns["result"])

@app.post("/")
def answer(req: Q):
    prompt = PROMPT.format(cols=list(DF.columns),
                           sample=DF.head(3).to_dict("records"),
                           statuses=DF["status"].unique().tolist(),
                           q=req.question)
    messages = [{"role": "user", "content": prompt}]
    for _ in range(2):
        code = ""
        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini", messages=messages,
                temperature=0, timeout=6)
            code = resp.choices[0].message.content
            return {"answer": run_code(code)}
        except Exception as e:
            messages += [{"role": "assistant", "content": code or "none"},
                         {"role": "user", "content": f"That failed: {e}. Fix it. Output only code."}]
    return {"answer": None}