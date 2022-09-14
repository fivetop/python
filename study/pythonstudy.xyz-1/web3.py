import requests

resp = requests.get('http://finance.naver.com/')
resp.raise_for_status()

resp.encoding = None  # None 으로 설정
# resp.encoding='euc-kr'  # 한글 인코딩

html = resp.text
print(html)