# test_api.py - Simple API test
import requests

# API URL
BASE_URL = "http://localhost:8000"

# 1. Upload a PDF
with open("multigen.pdf", "rb") as f:
    response = requests.post(
        f"{BASE_URL}/upload",
        files={"file": f}
    )
    print("Upload response:", response.json())
    session_id = response.json()["session_id"]

# 2. Ask a question
response = requests.post(
    f"{BASE_URL}/ask",
    json={
        "question": "What is this document about?",
        "session_id": session_id
    }
)
print("\nAnswer:", response.json())