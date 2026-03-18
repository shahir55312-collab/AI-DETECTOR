from fastapi import FastAPI
from pydantic import BaseModel
import json
import google.generativeai as genai
import re
import numpy as np
import os
app = FastAPI()
model = genai.GenerativeModel("gemini-2.5-flash")
api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key = api_key)
class Textanalyzer(BaseModel):
    text : str
def extract_json(text : str):
   match = re.search(r'\{.*\}', text, re.DOTALL)
   if match:       return match.group()
   else:       return None


def sentence_var(text : str):
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    return np.var([len(s) for s in sentences])


def word_diversity(text : str):
    words = text.lower().split()
    unique_words = set(words)
    return len(unique_words) / len(words) if words else 0
def repetition_score(text : str):
    words = text.lower().split()
    return len(words) - len(set(words)) if words else 0

async def analyze_confidence(text: str):
    prompt = f"""You are a confidence analyzer. Analyze the following text and determine the confidence level of the person speaking. Provide a score between 0 and 100, where 0 indicates no confidence and 100 indicates high confidence. Also, provide a brief explanation for your score.
    write it in this format :
    {{
    "ai_probability_score":number between 0 and 1,
    "confidence": Low|Medium|High,
    "reason" : your small reason for it 
    }}
    Text: {text}
    """



    response = model.generate_content(prompt)
    content  = response.text
    try:
        clean_json = extract_json(content)
        if clean_json:
            result = json.loads(clean_json)
            return result
        else:
            return {"error": "No valid JSON found in the model's response", "response": content}
    except:
        return {"error": "Failed to parse response from the model", "response": content}
    

def get_verdict(score : int):
    if score < 40:
        return "Human generated"
    elif score > 70:
        return "Ai generated"
    elif score >= 40 and score <= 70:
        return "Uncertain"
    else:
        return "Error"
    
@app.post("/analyze-confidence")
async def analyze_confidence_endpoint(text_analyzer: Textanalyzer):
    result = await analyze_confidence(text_analyzer.text)
    llm_score = result.get("ai_probability_score", None)
    varience = sentence_var(text_analyzer.text)
    diversity = word_diversity(text_analyzer.text)
    repetition = repetition_score(text_analyzer.text)
    
    diversity_score = 1 - diversity
    varience_score = min(varience / 50 , 1)
    repetition_score_n = min(repetition / 50 , 1)

    final_score = (
        llm_score * 0.6 + varience_score * 0.1 + diversity_score * 0.2 + repetition_score_n * 0.1
    )

    final_score = round(final_score,2)
    
    response = {
        "ai_probability_score": final_score,
        "llm_score": llm_score,
        "confidence" : result.get("confidence", "Unknown"),
        "reason" : result.get("reason", "No reason provided"),
        "verdict" : get_verdict(final_score * 100)
    }
    return response
