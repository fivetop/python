import matplotlib
import feedparser
import newspaper
from eunjeon import Mecab
from collections import Counter
import matplotlib.pyplot as plt
from wordcloud import WordCloud


def draw_wordcloud_from_rss(rss_link):
    #  feedparser, newspaper: RSS를 통해 뉴스의 본문을 수집
    feeds = feedparser.parse(rss_link)
    links = [entry['link'] for entry in feeds['entries']]

    news_text = ''
    for link in links:
        article = newspaper.Article(link, language='ko')
        article.download()
        article.parse()
        news_text += article.text

    # konlpy, Mecab: 형태소 분석을 통해 본문에서 명사추출, 1글자는 단어는 삭제
    engine = Mecab()
    nouns = engine.nouns(news_text)
    nouns = [n for n in nouns if len(n) > 1]

    # Counter: 단어수 세기, 가장 많이 등장한 단어(명사) 40개
    count = Counter(nouns)
    tags = count.most_common(20)

    # WordCloud, matplotlib: 단어 구름 그리기
    #font_path = '/usr/share/fonts/truetype/nanum/NanumMyeongjoBold.ttf'
    font_path = 'c:\\windows\\font\\gulim.ttc'
    wc = WordCloud(font_path=font_path, background_color='white', width=800, height=600)
    cloud = wc.generate_from_frequencies(dict(tags))
    plt.figure(figsize=(10, 8))
    plt.axis('off')
    plt.imshow(cloud)


# 경향신문 경제뉴스 RSS
rss_link = 'http://file.mk.co.kr/news/rss/rss_50300009.xml'
#draw_wordcloud_from_rss('http://www.khan.co.kr/rss/rssdata/itnews.xml')
draw_wordcloud_from_rss('http://rss.hankyung.com/new/news_main.xml')
#draw_wordcloud_from_rss(rss_link)