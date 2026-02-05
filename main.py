import csv
import time
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager

def collect_today_news_to_csv():
    url = "https://news.naver.com/breakingnews/section/105/226"

    options = Options()
    options.add_argument("--headless") # 서버 실행 필수
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")

    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    news_data = []
    seen_links = set()
    found_yesterday = False

    try:
        driver.get(url)
        while not found_yesterday:
            articles = driver.find_elements(By.CLASS_NAME, "sa_item")
            for article in articles:
                try:
                    dt_element = article.find_element(By.CSS_SELECTOR, ".sa_text_datetime b")
                    time_text = dt_element.text.strip()
                    if "1일전" in time_text:
                        found_yesterday = True
                        break
                    title_element = article.find_element(By.CLASS_NAME, "sa_text_title")
                    title = title_element.text.strip()
                    link = title_element.get_attribute("href")
                    if link not in seen_links:
                        news_data.append([title, time_text, link])
                        seen_links.add(link)
                except: continue
            if found_yesterday: break
            try:
                more_button = driver.find_element(By.CLASS_NAME, "section_more_inner")
                more_button.click()
                time.sleep(2)
            except: break
        save_to_csv(news_data)
    finally:
        driver.quit()

def save_to_csv(data):
    if not data: return
    with open("naver_today_news.csv", 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(['제목', '시간', '링크'])
        writer.writerows(data)

if __name__ == "__main__":
    collect_today_news_to_csv()
