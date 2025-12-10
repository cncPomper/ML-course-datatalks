import requests

url = "http://127.0.0.1:8000/predict"  # or YOUR_URL
client = {
    "lead_source": "organic_search",
    "number_of_courses_viewed": 4,
    "annual_income": 80304.0
}

resp = requests.post(url, json=client)
print(resp.status_code, resp.json())