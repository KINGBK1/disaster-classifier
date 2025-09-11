import os
from fastapi import FastAPI, HTTPException, Form, UploadFile, File
from google import genai
from google.genai import types

app = FastAPI()

# Initialize Gemma client
client = genai.Client(api_key = os.environ.get("GEMINI_API_KEY"))
model = "gemma-3-27b-it"

valid_labels = {"non_disaster", "low_risk", "mild_risk", "high_risk"}


def classify_disaster(post_text: str, image_bytes: bytes = None) -> str:
    """
    Classify disaster severity using Gemma, optionally with an image.
    Follows these rules:
    - Return exactly one label: non_disaster, low_risk, mild_risk, high_risk
    - Translate non-English posts before classification
    - If image is AI-generated or unrelated, output non_disaster
    - If uncertain, choose the least severe applicable label
    """

    prompt = f"""
You are a disaster severity classification system.

Task: Analyze the following post and return ONLY one label:
- non_disaster
- low_risk
- mild_risk
- high_risk

Rules:
- Respond with exactly one label (no explanation, no extra text).
- If it is not related to any disaster, output: non_disaster.
- If the language of the post is not English then first translate it to English and then classify it.
- If uncertain, choose the least severe applicable label.
- If image is AI generated or fake, output: non_disaster.
- If image is not related to disaster, output: non_disaster.

Post: "{post_text}"
    """

    parts = [types.Part.from_text(text=prompt)]

    if image_bytes:
        parts.append(
            types.Part.from_bytes(
                data=image_bytes,
                mime_type="image/jpeg"  # can also be image/png
            )
        )

    contents = [types.Content(role="user", parts=parts)]

    try:
        response = client.models.generate_content(
            model=model,
            contents=contents,
            config=types.GenerateContentConfig(
                temperature=0.0,
                max_output_tokens=5
            )
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gemma API error: {e}")

    prediction = response.text.strip().lower()
    if prediction not in valid_labels:
        prediction = "non_disaster"

    return prediction


@app.post("/predict")
async def predict(
    post: str = Form(...),
    image: UploadFile = File(None)  # optional image
):
    """
    Predict disaster severity for a given post text and optional image.
    Returns one of: non_disaster, low_risk, mild_risk, high_risk
    """
    image_bytes = await image.read() if image else None
    severity = classify_disaster(post, image_bytes)
    return {"severity": severity}


@app.get("/")
async def root():
    return {"message": "Disaster Classifier API running with Gemma!"}
