import csv
import time
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager

# 수집할 카테고리 정보 설정
CATEGORIES = {
    "모바일": "https://news.naver.com/breakingnews/section/105/731",
    "인터넷 & SNS": "https://news.naver.com/breakingnews/section/105/226",
    "통신 & 뉴미디어": "https://news.naver.com/breakingnews/section/105/227",
    "IT 일반": "https://news.naver.com/breakingnews/section/105/230",
    "컴퓨터": "https://news.naver.com/breakingnews/section/105/283",
    "과학 일반": "https://news.naver.com/breakingnews/section/105/228"
}

def setup_driver():
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
    return webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

def collect_section_news(driver, category_name, url):
    print(f"\n📂 [{category_name}] 섹션 수집 시작...")
    driver.get(url)
    
    news_data = []
    seen_links = set()
    found_yesterday = False

    while not found_yesterday:
        articles = driver.find_elements(By.CLASS_NAME, "sa_item")
        if not articles:
            break

        for article in articles:
            try:
                dt_element = article.find_element(By.CSS_SELECTOR, ".sa_text_datetime b")
                time_text = dt_element.text.strip()

                # "1일전" 발견 시 해당 섹션 수집 중단
                if "1일전" in time_text:
                    print(f"   ✋ '1일전' 기사 도달. [{category_name}] 수집 종료.")
                    found_yesterday = True
                    break

                title_element = article.find_element(By.CLASS_NAME, "sa_text_title")
                title = title_element.text.strip()
                link = title_element.get_attribute("href")

                if link not in seen_links:
                    # '분류' 컬럼을 추가하여 저장
                    news_data.append([category_name, title, time_text, link])
                    seen_links.add(link)
            except:
                continue

        if found_yesterday:
            break

        # '더보기' 버튼 클릭 처리
        try:
            more_button = driver.find_element(By.CLASS_NAME, "section_more_inner")
            if more_button.is_displayed():
                more_button.click()
                time.sleep(1.5)
            else:
                break
        except:
            break
            
    return news_data

def save_to_csv(all_data):
    if not all_data:
        print("\n❌ 수집된 데이터가 없어 파일을 생성하지 않습니다.")
        return

    filename = "naver_today_news.csv"
    header = ['분류', '제목', '시간', '링크']

    with open(filename, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(all_data)
    print(f"\n💾 저장 완료: {filename} (총 {len(all_data)}건)")

if __name__ == "__main__":
    driver = setup_driver()
    total_news = []

    try:
        for category, url in CATEGORIES.items():
            section_data = collect_section_news(driver, category, url)
            total_news.extend(section_data)
            
        save_to_csv(total_news)
    finally:
        driver.quit()
