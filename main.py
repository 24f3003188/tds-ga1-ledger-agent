from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
import pandas as pd
import pytz
import os
import io
from openai import OpenAI

app = FastAPI()

client = OpenAI(
    api_key=os.environ.get("AIPIPE_TOKEN"),
    base_url="https://aipipe.org/openai/v1"
)

# Your specific links
EXPORT_URL = "https://exam.sanand.workers.dev/questionData?email=24f3003188%40ds.study.iitm.ac.in&quizSign=0eyEF2vP3oDh3px4Vo9tHq71IUREcIBMw1EaaKOnFW%2B1UjL9P9wdpb%2BeSpssg%2BTBBUrwhwiNQq6eP7rfINIEW02t%2BV2%2B6VRmhVnmIR1bUNbnmebNkceN9GqbctO9qMtZM34SaH835zl934rGiZU2buVnN4k0ukAhSOgidNU937swfeIxT4C7DGTXSlKOixvebm%2BogWfa428bTAnEjTWD0Anq74Swn8jsnnq3qaNQm4tKka%2BfUxh8M3iq5kS4h%2FX1A3ONm1qNCal6lJxKAP%2FuTWM94UXVF8n9GHUdQ72GmNL5jBEudtFnRyFiECqe5%2B%2FopPut9MsHAywOHqoxDwCwgQ%3D%3D&questionId=q-ledger-agent-server&path=%2Fexport"
RATES_URL = "https://exam.sanand.workers.dev/questionData?email=24f3003188%40ds.study.iitm.ac.in&quizSign=0eyEF2vP3oDh3px4Vo9tHq71IUREcIBMw1EaaKOnFW%2B1UjL9P9wdpb%2BeSpssg%2BTBBUrwhwiNQq6eP7rfINIEW02t%2BV2%2B6VRmhVnmIR1bUNbnmebNkceN9GqbctO9qMtZM34SaH835zl934rGiZU2buVnN4k0ukAhSOgidNU937swfeIxT4C7DGTXSlKOixvebm%2BogWfa428bTAnEjTWD0Anq74Swn8jsnnq3qaNQm4tKka%2BfUxh8M3iq5kS4h%2FX1A3ONm1qNCal6lJxKAP%2FuTWM94UXVF8n9GHUdQ72GmNL5jBEudtFnRyFiECqe5%2B%2FopPut9MsHAywOHqoxDwCwgQ%3D%3D&questionId=q-ledger-agent-server&path=%2Frates"

class QuestionRequest(BaseModel):
    question: str

# Cache the data so we don't redownload it for every single question
_data_cache = None
_rates_cache = None

def get_data():
    global _data_cache, _rates_cache
    if _data_cache is not None:
        return _data_cache, _rates_cache

    try:
        # Fetch rates
        _rates_cache = requests.get(RATES_URL).json()

        # Fetch export and parse JSONL (JSON Lines)
        res = requests.get(EXPORT_URL)
        df = pd.read_json(io.StringIO(res.text), lines=True)

        # Rule: Deduplicate based on 'id' keeping the latest 'updated_at'
        df["updated_at"] = pd.to_datetime(df["updated_at"])
        df = df.sort_values("updated_at").drop_duplicates(subset=["id"], keep="last")
        
        # Ensure created_at is a datetime object in Kolkata timezone
        df["created_at"] = pd.to_datetime(df["created_at"])

        _data_cache = df
        return _data_cache, _rates_cache
    except Exception as e:
        print(f"Data fetch error: {e}")
        return pd.DataFrame(), {}

@app.post("/")
def answer_question(req: QuestionRequest):
    df, rates = get_data()
    if df.empty:
        raise HTTPException(status_code=500, detail="Failed to load ledger data.")

    # We ask the AI to generate a Pandas command that computes the answer
    prompt = f"""
    You are a data analyst. I have a Pandas DataFrame `df`.
    Columns: {list(df.columns)}
    Sample row: {df.iloc[0].to_dict()}
    
    Currency Exchange Rates `rates`: {rates}

    Rules:
    - Only orders with status 'paid' count as revenue/valid unless asking about refunds.
    - Money must be answered in USD. You MUST use the `rates` dictionary to convert the `amount` column from the local `currency` to USD.
    - Count unique customers using `nunique()`.

    Question: "{req.question}"

    Write ONLY valid Python code using pandas that computes the answer and assigns it to a variable named `result`.
    Do not use markdown backticks (no ```python). Just the raw python code.
    Example output format:
    result = df[(df['status'] == 'paid') & (df['product'] == 'Rice Cooker')]['customer'].nunique()
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        
        # Clean up the AI's output in case it includes markdown
        code = response.choices[0].message.content.strip().replace("```python", "").replace("```", "")
        
        # Execute the AI's code safely in memory
        local_vars = {"df": df, "rates": rates, "pd": pd}
        exec(code, {}, local_vars)
        ans = local_vars.get("result")

        # Format numeric output (correct to the cent)
        if isinstance(ans, (float, int)):
            return {"answer": round(float(ans), 2)}
        return {"answer": ans}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Code generation error: {str(e)}\nGenerated Code: {code}")