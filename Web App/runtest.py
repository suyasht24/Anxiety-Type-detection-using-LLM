import requests

url = "http://127.0.0.1:5000/predict"
data = {"text": "I feel anxious in social situations."}
response = requests.post(url, json=data)

print(response.json())
