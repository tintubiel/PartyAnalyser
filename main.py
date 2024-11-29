import requests

url = "http://0.0.0.0:2444/predict"
image_path = '/Users/tintubiel/Desktop/images.jpeg'

with open(image_path, 'rb') as image_file:
    response = requests.post(url, files={'image': image_file})

print(response.json())
