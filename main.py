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

# 1. 수집할 카테고리 설정
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
    """유사도 측정을 위해 한글/공백만 남김"""
    return re.sub(r'[^가-힣\s]', '', text)

def filter_ai_keywords(data_list):
    """제목에 'AI', 'ai', '인공지능'이 포함된 기사만 1차 추출"""
    filtered_data = []
    pattern = re.compile(r'ai|인공지능', re.IGNORECASE)
    
    for item in data_list:
        title = item[1]
        if pattern.search(title):
            filtered_data.append(item)
    return filtered_data

def deduplicate_articles(data_list, threshold=0.2):
    """추출된 AI 기사들 중 유사한 제목 제거 (기준 0.2)"""
    if not data_list:
        return []

    df = pd.DataFrame(data_list, columns=['분류', '제목', '시간', '링크'])
    final_indices = []

    for category in df['분류'].unique():
        category_df = df[df['분류'] == category].copy()
        if len(category_df) <= 1:
            final_indices.extend(category_df.index.tolist())
            continue

        titles = category_df['제목'].apply(clean_text).tolist()
        vectorizer = TfidfVectorizer(min_df=1)
        tfidf_matrix = vectorizer.fit_transform(titles)
        cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
        
        keep_mask = [True] * len(category_df)
        for i in range(len(category_df)):
            if not keep_mask[i]: continue
            for j in range(i + 1, len(category_df)):
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
            more_button.click()
            time.sleep(1.5)
        except: break
            
    return news_data

def save_to_csv(all_data):
    if not all_data:
        print("\n❌ 최종 결과가 없어 파일을 생성하지 않습니다.")
        return

    filename = "naver_today_news.csv"
    with open(filename, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(['분류', '제목', '시간', '링크'])
        writer.writerows(all_data)
    print(f"\n💾 저장 완료: {filename} (최종 {len(all_data)}건)")

if __name__ == "__main__":
    driver = setup_driver()
    raw_news = []

    try:
        # 1. 뉴스 전체 수집
        for category, url in CATEGORIES.items():
            raw_news.extend(collect_section_news(driver, category, url))
        
        print(f"\n--- 1단계: 수집 완료 ({len(raw_news)}건) ---")
        
        # 2. AI 관련 기사 1차 필터링 (순서 변경됨)
        print("🔍 2단계: AI 관련 기사 추출 중...")
        ai_news = filter_ai_keywords(raw_news)
        
        # 3. 필터링된 결과 내에서 중복 제거 (threshold 0.2)
        print(f"🤖 3단계: 유사도 기반 중복 제거 중 (기준: 0.2, 대상: {len(ai_news)}건)...")
        final_news = deduplicate_articles(ai_news, threshold=0.2)
        
        print(f"\n✨ 최종 요약: 전체({len(raw_news)}건) -> AI추출({len(ai_news)}건) -> 중복제거({len(final_news)}건)")
        
        # 4. 저장
        save_to_csv(final_news)
        
    finally:
        driver.quit()
