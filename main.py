import csv
import time
import re
import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

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

def clean_text(text):
    """제목에서 한글과 공백만 남기고 제거 (유사도 측정 정확도 향상)"""
    return re.sub(r'[^가-힣\s]', '', text)

def deduplicate_articles(data_list, threshold=0.4):
    """TF-IDF와 코사인 유사도를 이용한 중복 기사 제거"""
    if not data_list:
        return []

    df = pd.DataFrame(data_list, columns=['분류', '제목', '시간', '링크'])
    final_indices = []

    # 각 카테고리별로 독립적으로 중복 체크 수행
    for category in df['분류'].unique():
        category_df = df[df['분류'] == category].copy()
        if len(category_df) <= 1:
            final_indices.extend(category_df.index.tolist())
            continue

        # 1. 텍스트 정제 및 벡터화
        titles = category_df['제목'].apply(clean_text).tolist()
        vectorizer = TfidfVectorizer(min_df=1)
        tfidf_matrix = vectorizer.fit_transform(titles)
        
        # 2. 코사인 유사도 계산
        cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
        
        # 3. 유사도 기반 필터링
        keep_mask = [True] * len(category_df)
        for i in range(len(category_df)):
            if not keep_mask[i]: continue
            for j in range(i + 1, len(category_df)):
                # 설정한 threshold(0.4)보다 높으면 중복으로 간주
                if cosine_sim[i, j] > threshold:
                    keep_mask[j] = False
        
        final_indices.extend(category_df.iloc[keep_mask].index.tolist())

    return df.loc[final_indices].values.tolist()

def collect_section_news(driver, category_name, url):
    print(f"\n📂 [{category_name}] 섹션 수집 시작...")
    driver.get(url)
    news_data = []
    seen_links = set()
    found_yesterday = False

    while not found_yesterday:
        articles = driver.find_elements(By.CLASS_NAME, "sa_item")
        if not articles: break

        for article in articles:
            try:
                dt_element = article.find_element(By.CSS_SELECTOR, ".sa_text_datetime b")
                time_text = dt_element.text.strip()

                if "1일전" in time_text:
                    print(f"   ✋ '1일전' 기사 도달. [{category_name}] 종료.")
                    found_yesterday = True
                    break

                title_element = article.find_element(By.CLASS_NAME, "sa_text_title")
                title = title_element.text.strip()
                link = title_element.get_attribute("href")

                if link not in seen_links:
                    news_data.append([category_name, title, time_text, link])
                    seen_links.add(link)
            except: continue

        if found_yesterday: break

        try:
            more_button = driver.find_element(By.CLASS_NAME, "section_more_inner")
            if more_button.is_displayed():
                more_button.click()
                time.sleep(1.5)
            else: break
        except: break
            
    return news_data

def save_to_csv(all_data):
    if not all_data:
        print("\n❌ 저장할 데이터가 없습니다.")
        return

    filename = "naver_today_news.csv"
    with open(filename, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(['분류', '제목', '시간', '링크'])
        writer.writerows(all_data)
    print(f"\n💾 저장 완료: {filename} (총 {len(all_data)}건)")

if __name__ == "__main__":
    driver = setup_driver()
    raw_news = []

    try:
        # 1. 모든 카테고리 순회하며 수집
        for category, url in CATEGORIES.items():
            raw_news.extend(collect_section_news(driver, category, url))
        
        print(f"\n--- 수집 완료 (총 {len(raw_news)}건) ---")
        
        # 2. 자연어 처리로 유사 제목 제거
        print("🤖 AI 중복 필터링 작동 중...")
        filtered_news = deduplicate_articles(raw_news, threshold=0.4)
        print(f"✨ 필터링 결과: {len(raw_news)}건 -> {len(filtered_news)}건으로 압축")
        
        # 3. 최종 결과 저장
        save_to_csv(filtered_news)
        
    finally:
        driver.quit()
