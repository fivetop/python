# -*- coding: utf-8 -*-

import requests, bs4

resp = requests.get('http://finance.naver.com/')
resp.raise_for_status()
print(resp.text)

resp.encoding = 'euc-kr'
html = resp.text

bs = bs4.BeautifulSoup(html, 'html.parser')
tags = bs.select('div.news_area h2 a')  # Top 뉴스
#title = tags[0].getText()
#print(title)
print(tags)