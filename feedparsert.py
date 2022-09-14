import feedparser

#  경향닷컴 경제뉴스 RSS
feeds = feedparser.parse('http://www.khan.co.kr/rss/rssdata/economy.xml')
links = [entry['link'] for entry in feeds['entries']]
links