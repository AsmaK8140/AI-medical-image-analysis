import requests

url = "http://127.0.0.1:5000/predict"

files = {
    "image": open("data/chest_xray/test/PNEUMONIA/person1_virus_6.jpeg", "rb")
}

response = requests.post(url, files=files)

print(response.json())