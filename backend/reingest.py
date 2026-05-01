import requests

paper_id = "09aa04bf6edf4481b696d0d9179ede2b"
url = f"http://localhost:8000/api/papers/{paper_id}/re-ingest"

response = requests.post(url)
if response.status_code == 200:
    print("Re-ingested successfully")
    print(response.json())
else:
    print(f"Error: {response.status_code}")
    print(response.text)