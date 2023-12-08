import requests
from requests.auth import HTTPDigestAuth

URL = 'http://localhost:5000'
id = 'hza'

# urlk
def send_image(image_path):
    with requests.Session() as s:
        out = s.post(URL + '/auth/login', data={'username': 'hza', 'password': '123'})
        res = s.post(URL + '/create', data={'image_path': image_path, 'id': id, 'title': 'image', 'body': 'image'})
        return res.text