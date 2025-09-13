import os, re
from fastapi import FastAPI, HTTPException, Form, UploadFile, File
from google import genai
from google.genai import types

app = FastAPI()

# Initialize Gemini client
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
model = "gemini-2.0-flash"

valid_labels = {"non_disaster", "low_risk", "mild_risk", "high_risk"}


def classify_disaster(post_text: str, image_bytes: bytes = None, mime_type: str = None) -> str:
    """Classify disaster severity using Gemini multimodal API."""

    prompt = f"""
    You are a disaster severity classification system.

    Task: Analyze the following post (and optional image) and return ONLY one label:
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

    if image_bytes and mime_type:
        parts.append(
            types.Part.from_bytes(
                data=image_bytes,
                mime_type=mime_type
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
        raise HTTPException(status_code=500, detail=f"Gemini API error: {e}")

    prediction = response.text.strip().lower()

    # Safer cleanup: extract only valid label
    match = re.search(r"(non_disaster|low_risk|mild_risk|high_risk)", prediction)
    return match.group(1) if match else "non_disaster"


@app.post("/predict")
async def predict(
    post: str = Form(...),
    image: UploadFile = File(None)  # optional image upload
):
    """
    Predict disaster severity for a given post text and optional image.
    Returns one of: non_disaster, low_risk, mild_risk, high_risk
    """
    image_bytes = await image.read() if image else None
    mime_type = image.content_type if image else None
    severity = classify_disaster(post, image_bytes, mime_type)
    return {"severity": severity}


@app.get("/")
async def root():
    return {"message": "Disaster Classifier API running with Gemini 2.0 Flash!"}

@app.get("/ping")
def ping():
    return {"status": "ok", "message": "FastAPI server is awake"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=4000)
