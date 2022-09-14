import requests

resp = requests.get('http://daum.net')
# resp.raise_for_status()

if (resp.status_code == requests.codes.ok):
    html = resp.text
    print(html)