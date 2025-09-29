from dotenv import load_dotenv
import os

load_dotenv()  # loads .env
print(os.getenv("GROQ_API_KEY"))  # optional test
