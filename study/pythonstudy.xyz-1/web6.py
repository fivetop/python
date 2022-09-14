from selenium import webdriver
import time

# browser = webdriver.Firefox()
browser = webdriver.Chrome()
browser.get("http://python.org")

menus = browser.find_elements_by_css_selector('#top ul.menu li')

pypi = None
for m in menus:
    if m.text == "PyPI":
        pypi = m
    print(m.text)

pypi.click()  # 클릭

time.sleep(5)  # 5초 대기
browser.quit()  # 브라우저 종료
