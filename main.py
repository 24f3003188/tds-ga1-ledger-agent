from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
import pandas as pd
import pytz
import os
from openai import OpenAI

# 1. Initialize FastAPI app first
app = FastAPI()

# 2. Configure OpenAI client to use AI Pipe
client = OpenAI(
    api_key=os.environ.get("AIPIPE_TOKEN"),
    base_url="https://aipipe.org/openai/v1"
)

EXPORT_URL = "https://exam.sanand.workers.dev/questionData?email=24f3003188%40ds.study.iitm.ac.in&quizSign=0eyEF2vP3oDh3px4Vo9tHq71IUREcIBMw1EaaKOnFW%2B1UjL9P9wdpb%2BeSpssg%2BTBBUrwhwiNQq6eP7rfINIEW02t%2BV2%2B6VRmhVnmIR1bUNbnmebNkceN9GqbctO9qMtZM34SaH835zl934rGiZU2buVnN4k0ukAhSOgidNU937swfeIxT4C7DGTXSlKOixvebm%2BogWfa428bTAnEjTWD0Anq74Swn8jsnnq3qaNQm4tKka%2BfUxh8M3iq5kS4h%2FX1A3ONm1qNCal6lJxKAP%2FuTWM94UXVF8n9GHUdQ72GmNL5jBEudtFnRyFiECqe5%2B%2FopPut9MsHAywOHqoxDwCwgQ%3D%3D&questionId=q-ledger-agent-server&path=%2Fexport"

class QuestionRequest(BaseModel):
    question: str

def get_cleaned_data():
    res = requests.get(EXPORT_URL)
    data = res.json()
    
    df = pd.DataFrame(data if isinstance(data, list) else data.get("orders", []))
    if df.empty:
        return df

    # Rule: Latest updated_at for an order_id is current
    if "updated_at" in df.columns and "order_id" in df.columns:
        df["updated_at"] = pd.to_datetime(df["updated_at"])
        df = df.sort_values("updated_at").drop_duplicates(subset=["order_id"], keep="last")

    # Rule: Business dates use Asia/Kolkata timezone
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])

    return df

@app.post("/")
def answer_question(req: QuestionRequest):
    df = get_cleaned_data()
    
    prompt = f"""
    You are a financial data assistant for Acme Appliances.
    Here is the cleaned ledger data (as JSON records):
    {df.to_json(orient='records')}

    Rules:
    - Only orders with status 'paid' count as revenue.
    - Business dates use Asia/Kolkata timezone.
    - Money must be in USD, correct to the cent.
    - Product names must match the API spelling.

    Question: "{req.question}"

    Return ONLY the final answer (a plain number, exact text, or product name). No extra commentary.
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    
    ans = response.choices[0].message.content.strip()

    try:
        if "." in ans:
            return {"answer": float(ans)}
        return {"answer": int(ans)}
    except ValueError:
        return {"answer": ans}