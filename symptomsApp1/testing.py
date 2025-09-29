import os
from groq import Groq
import traceback

# -------------------------------
# Initialize Groq client
# -------------------------------
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise Exception("GROQ_API_KEY not set in environment variables or .env file")

"""client = Groq(api_key=api_key)
print("Groq client initialized successfully")

# -------------------------------
# Function to get friendly advice
# -------------------------------
def get_friendly_advice(disease_name):
    
    try:
        resp = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a helpful medical assistant."},
                {"role": "user", "content": f"Explain in simple words what {disease_name} is and give friendly advice."}
            ],
            model="qwen/qwen3-32b",
            temperature=0.5,
            max_tokens=400,          # use max_tokens with chat.completions
            stream=False
        )
        output = resp.choices[0].message.content.strip()
        if not output:
            return f"No advice available for: {disease_name}"
        return output

    except Exception:
        print("Groq API error:")
        traceback.print_exc()
        return f"Unable to fetch advice for: {disease_name}"

# -------------------------------
# Quick test
# -------------------------------
if __name__ == "__main__":
    disease = "diabetes"
    advice = get_friendly_advice(disease)
    print("Advice:", advice)"""
import os
from groq import Groq
import traceback

# -------------------------------
# Initialize Groq client
# -------------------------------
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise Exception("GROQ_API_KEY not set in environment variables or .env file")

client = Groq(api_key=api_key)
print("Groq client initialized successfully")

# -------------------------------
# Function to get friendly advice
# -------------------------------
def get_friendly_advice(disease_name):
    """
    Returns human-friendly advice for a disease using Groq Llama model.
    """
    try:
        resp = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a helpful medical assistant."},
                {"role": "user", "content": f"Explain in simple words what {disease_name} is and give friendly advice."}
            ],
            model="llama-3.3-70b-versatile",  # Groq model ID
            temperature=0.5,
            max_tokens=400,                   # compatible with current SDKs
            stream=False
        )
        output = resp.choices[0].message.content.strip()
        if not output:
            return f"No advice available for: {disease_name}"
        return output

    except Exception:
        print("Groq API error:")
        traceback.print_exc()
        return f"Unable to fetch advice for: {disease_name}"

# -------------------------------
# Quick test
# -------------------------------
if __name__ == "__main__":
    disease = "diabetes"
    advice = get_friendly_advice(disease)
    print("Advice:", advice)
