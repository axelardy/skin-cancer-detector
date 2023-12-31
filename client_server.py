import requests

url = 'http://140.138.172.215:5000/predict'

with open('ISIC_0024461.jpg', 'rb') as f:
    r = requests.post(url, files={'image': f})

    print(r.json())