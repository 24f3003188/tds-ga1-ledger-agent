
import os
import json
import re

import requests
import pandas as pd
import numpy as np

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import OpenAI


ROOT = (
    "https://exam.sanand.workers.dev/questionData"
    "?email=24f3003188%40ds.study.iitm.ac.in"
    "&quizSign=0eyEF2vP3oDh3px4Vo9tHq71IUREcIBMw1EaaKOnFW%2B1UjL9P9wdpb%2BeSpssg%2BTBBUrwhwiNQq6eP7rfINIEW02t%2BV2%2B6VRmhVnmIR1bUNbnmebNkceN9GqbctO9qMtZM34SaH835zl934rGiZU2buVnN4k0ukAhSOgidNU937swfeIxT4C7DGTXSlKOixvebm%2BogWfa428bTAnEjTWD0Anq74Swn8jsnnq3qaNQm4tKka%2BfUxh8M3iq5kS4h%2FX1A3ONm1qNCal6lJxKAP%2FuTWM94UXVF8n9GHUdQ72GmNL5jBEudtFnRyFiECqe5%2B%2FopPut9MsHAywOHqoxDwCwgQ%3D%3D"
    "&questionId=q-ledger-agent-server&path=/"
)

TZ = "Asia/Kolkata"

app = FastAPI()

client = OpenAI(
    api_key=os.environ["AIPIPE_TOKEN"],
    base_url="https://aipipe.org/openai/v1",
)


class Q(BaseModel):
    question: str


def load():
    response = requests.get(ROOT, timeout=20)
    response.raise_for_status()
    links = response.json()["links"]

    response = requests.get(links["rates"], timeout=20)
    response.raise_for_status()
    rates = response.json()["usd_per_unit"]

    response = requests.get(links["export"], timeout=30)
    response.raise_for_status()

    df = pd.DataFrame(
        json.loads(line)
        for line in response.text.splitlines()
        if line.strip()
    )

    # Keep the latest version of each order.
    df["updated_at"] = pd.to_datetime(
        df["updated_at"], utc=True, format="ISO8601"
    )
    df = (
        df.sort_values("updated_at")
        .drop_duplicates("id", keep="last")
        .copy()
    )

    # Normalize status values.
    df["status"] = df["status"].astype(str).str.strip().str.lower()

    # Convert timestamps to the business timezone.
    ts = pd.to_datetime(
        df["created_at"], utc=True, format="ISO8601"
    ).dt.tz_convert(TZ)

    df["month"] = ts.dt.strftime("%Y-%m")
    df["date"] = ts.dt.strftime("%Y-%m-%d")

    # Convert every order amount to USD exactly once.
    df["usd"] = (
        df["amount"].astype(float)
        * df["currency"].map(rates).astype(float)
    )

    df = df.drop(columns=["created_at", "updated_at"])
    return df.reset_index(drop=True), rates


DF, RATES = load()


PROMPT = """
You write pandas code. A DataFrame named df already exists.

Columns: {cols}
Sample rows: {sample}
Statuses in the data: {statuses}
Exact product names: {products}
Exact region names: {regions}

Rules:
- Return ONLY executable Python code, without Markdown fences.
- Store the final answer in a variable named result.
- result must be a plain number or string.
- Do not print or round the answer.
- Never modify the original data intentionally.

Money:
- df["usd"] is already converted to USD. Never convert it again.
- Revenue is the sum of usd for rows where status == "paid".
- Refunds are rows where status == "refunded".
- Refund amount is the sum of usd for refunded rows.
- Void rows do not count toward revenue, units sold, or refunds.

Dates:
- month is formatted as YYYY-MM.
- date is formatted as YYYY-MM-DD.
- Both already use the Asia/Kolkata business timezone.
- Filter using month and date, not the original timestamps.

Customers:
- Count distinct customers using df["customer"].nunique().
- Apply ALL question filters before counting unique customers.
- "Different customers", "unique customers", and "how many
  customers" require a distinct customer count, not an order count.
- For customers who placed at least one paid order for a product,
  filter status == "paid" AND product == the exact product name,
  then count df["customer"].nunique().
- Do not count refunded or void orders as paid orders.
- Do not count customers from unrelated products or regions.

Products:
- Match product names and region names exactly as listed above.
- Top-selling by revenue means groupby("product")["usd"].sum().idxmax().
- Top-selling by units means groupby("product")["qty"].sum().idxmax().
- Return product names exactly as written in the data.

Examples:

Q: Total revenue in USD from the North region in March 2026?
result = df[(df["status"] == "paid") & (df["region"] == "North") & (df["month"] == "2026-03")]["usd"].sum()

Q: Which product had the most revenue in April 2026?
result = df[(df["status"] == "paid") & (df["month"] == "2026-04")].groupby("product")["usd"].sum().idxmax()

Q: How many different customers placed at least one paid order for a Rice Cooker?
result = df[(df["status"] == "paid") & (df["product"] == "Rice Cooker")]["customer"].nunique()

Question: {q}
"""


def to_json_safe(x):
    if isinstance(x, pd.DataFrame):
        x = x.iloc[0, 0]

    if isinstance(x, (pd.Series, pd.Index)):
        x = x.iloc[0] if len(x) == 1 else x.tolist()

    if isinstance(x, np.generic):
        x = x.item()

    if isinstance(x, float) and not np.isfinite(x):
        return None

    if isinstance(x, float):
        x = round(x, 2)

    if isinstance(x, (str, int, float, bool)) or x is None:
        return x

    if isinstance(x, list):
        return [to_json_safe(v) for v in x]

    return str(x)


def direct_customer_count(question):
    """
    Handle unique-customer questions about a named product
    without depending on LLM-generated pandas code.
    """
    q = question.lower()

    customer_question = (
        "customer" in q
        and any(word in q for word in (
            "how many", "count", "number of",
            "different", "unique"
        ))
    )

    product_question = "product" in DF.columns and any(
        word in q for word in ("order", "orders", "bought", "purchase")
    )

    if not (customer_question and product_question):
        return None, False

    # Match a product name from the actual ledger, not a guessed name.
    matched_product = None
    for product in sorted(
        DF["product"].dropna().astype(str).unique(),
        key=len,
        reverse=True,
    ):
        if product.lower() in q:
            matched_product = product
            break

    if matched_product is None:
        return None, False

    # Only use this shortcut when the question explicitly asks
    # about paid orders.
    paid_question = any(
        phrase in q
        for phrase in ("paid", "payment received", "successfully paid")
    )

    if not paid_question:
        return None, False

    subset = DF[
        (DF["status"] == "paid")
        & (DF["product"] == matched_product)
    ]

    # A customer counts once, regardless of their order count.
    return int(subset["customer"].nunique()), True


def run_code(code):
    code = re.sub(
        r"^\s*```(?:python)?\s*|\s*```\s*$",
        "",
        code.strip(),
        flags=re.IGNORECASE,
    )

    ns = {
        "df": DF.copy(),
        "rates": RATES,
        "pd": pd,
        "np": np,
    }

    exec(code, ns)
    if "result" not in ns:
        raise ValueError("Generated code did not define result")

    return to_json_safe(ns["result"])


@app.post("/")
def answer(req: Q):
    question = req.question.strip()

    if not question:
        raise HTTPException(status_code=400, detail="Question is empty")

    # Deterministic solution for product-specific customer counts.
    direct_result, handled = direct_customer_count(question)
    if handled:
        return {"answer": direct_result}

    prompt = PROMPT.format(
        cols=list(DF.columns),
        sample=DF.head(3).to_dict("records"),
        statuses=DF["status"].unique().tolist(),
        products=DF["product"].dropna().unique().tolist(),
        regions=DF["region"].dropna().unique().tolist(),
        q=question,
    )

    messages = [{"role": "user", "content": prompt}]
    last_error = None

    for _ in range(2):
        code = ""
        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                temperature=0,
                timeout=8,
            )

            code = resp.choices[0].message.content or ""
            result = run_code(code)
            return {"answer": result}

        except Exception as e:
            last_error = str(e)
            messages.extend([
                {"role": "assistant", "content": code or "none"},
                {
                    "role": "user",
                    "content": (
                        f"That failed: {last_error}. "
                        "Correct the Python code. Output only code."
                    ),
                },
            ])

    raise HTTPException(
        status_code=500,
        detail=f"Could not answer the question: {last_error}",
    )