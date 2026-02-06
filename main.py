import csv
import time
import re
import os
import datetime
import pandas as pd
from google import genai
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- 1. 환경 설정 및 카테고리 ---
CATEGORIES = {
    "모바일": "https://news.naver.com/breakingnews/section/105/731",
    "인터넷 & SNS": "https://news.naver.com/breakingnews/section/105/226",
    "통신 & 뉴미디어": "https://news.naver.com/breakingnews/section/105/227",
    "IT 일반": "https://news.naver.com/breakingnews/section/105/230",
    "컴퓨터": "https://news.naver.com/breakingnews/section/105/283",
    "과학 일반": "https://news.naver.com/breakingnews/section/105/228"
}

# Gemini API 설정 (GitHub Secrets 혹은 환경변수에서 로드)
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

def setup_driver():
    options = Options()
    options.add_argument("--headless") # GitHub Actions 실행 시 필수
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
    return webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

# --- 2. 수집 및 정제 엔진 ---

def clean_text(text):
    return re.sub(r'[^가-힣\s]', '', text)

def collect_section_news(driver, category_name, url):
    print(f"📂 [{category_name}] 섹션 수집 시작...")
    driver.get(url)
    news_data, seen_links, found_yesterday = [], set(), False

    while not found_yesterday:
        articles = driver.find_elements(By.CLASS_NAME, "sa_item")
        if not articles: break

        for article in articles:
            try:
                dt_el = article.find_element(By.CSS_SELECTOR, ".sa_text_datetime b")
                time_text = dt_el.text.strip()
                if "1일전" in time_text:
                    found_yesterday = True
                    break

                title_el = article.find_element(By.CLASS_NAME, "sa_text_title")
                title, link = title_el.text.strip(), title_el.get_attribute("href")

                if link not in seen_links:
                    news_data.append([category_name, title, time_text, link])
                    seen_links.add(link)
            except: continue

        if found_yesterday: break
        try:
            more_btn = driver.find_element(By.CLASS_NAME, "section_more_inner")
            more_btn.click()
            time.sleep(1.5)
        except: break
    return news_data

def filter_ai_keywords(data_list):
    pattern = re.compile(r'ai|인공지능', re.IGNORECASE)
    return [item for item in data_list if pattern.search(item[1])]

def deduplicate_articles(data_list, threshold=0.2):
    if not data_list: return []
    df = pd.DataFrame(data_list, columns=['분류', '제목', '시간', '링크'])
    final_indices = []
    for category in df['분류'].unique():
        cat_df = df[df['분류'] == category].copy()
        if len(cat_df) <= 1:
            final_indices.extend(cat_df.index.tolist()); continue
        titles = cat_df['제목'].apply(clean_text).tolist()
        matrix = TfidfVectorizer().fit_transform(titles)
        sim = cosine_similarity(matrix, matrix)
        keep = [True] * len(cat_df)
        for i in range(len(cat_df)):
            if not keep[i]: continue
            for j in range(i+1, len(cat_df)):
                if sim[i, j] > threshold: keep[j] = False
        final_indices.extend(cat_df.iloc[keep].index.tolist())
    return df.loc[final_indices].values.tolist()

# --- 3. Gemini 3 지능형 분석 엔진 ---

def analyze_category_with_gemini(category_name, articles):
    if not articles:
        return f"<h3>{category_name}</h3><p>수집된 주요 AI 뉴스가 없습니다.</p>"

    article_list_str = "\n".join([f"- {a[1]} ({a[3]})" for a in articles])

    prompt = f"""
    당신은 IT 전문 분석가입니다. 아래 제공된 '{category_name}' 분야의 뉴스들을 구글 검색으로 확인하고 정독한 뒤 리포트를 작성하세요.

    [분석 제외] AI 비핵심 기사, 단순 주가/시총 뉴스, 정보 없는 일반 인사이트 기사.
    [작성 규칙]
    1. 가장 많이 언급되는 핵심 이슈 요약: 현재 해당 분야의 가장 큰 흐름을 2문장으로 요약하고 관련 링크를 제공
    2. 신제품/신기능 소식: AI 관련 신제품, 신기능, 서비스 출시 및 예정 소식이 있다면 최대 3문장으로 요약
    3.사회/제도/시장의 변화: AI로 인한 기존 시스템이나 시장 구조의 구체적인 '변화' 내용을 요약
    4. **[필수] 전문 용어는 괄호를 사용해 친절하게 풀어서 설명할 것.**

    기사 리스트:
    {article_list_str}
    """


    try:
        print(f"🤖 Gemini 3 분석 중: {category_name}")
        response = client.models.generate_content(
            model='gemini-2.0-flash-exp', # 최신 모델명 확인 필요
            contents=prompt,
            config={'tools': [{'google_search': {}}]}
        )
        return f"<section><h2>📌 {category_name} 동향 분석</h2>{response.text}</section><hr>"
    except Exception as e:
        return f"<section><h2>📌 {category_name} 분석 에러</h2><p>{e}</p></section><hr>"

# --- 4. HTML 생성 및 메인 실행 ---

def generate_html_report(report_body):
    now_kst = (datetime.datetime.utcnow() + datetime.timedelta(hours=9)).strftime('%Y-%m-%d %H:%M')
    
    html_template = f"""
    <!DOCTYPE html>
    <html lang="ko">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Daily AI Report - {now_kst}</title>
        <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/@picocss/pico@1/css/pico.min.css">
        <style>
            body {{ padding: 20px; max-width: 900px; margin: auto; }}
            h1 {{ color: #1095c1; text-align: center; margin-bottom: 50px; }}
            h2 {{ color: #1a1a1a; background: #f4f4f4; padding: 10px; border-radius: 8px; }}
            .update-time {{ text-align: right; font-size: 0.9rem; color: #666; }}
            section {{ margin-bottom: 40px; }}
            hr {{ margin: 40px 0; border: 0; border-top: 1px solid #eee; }}
        </style>
    </head>
    <body>
        <header>
            <p class="update-time">마지막 업데이트: {now_kst} (KST)</p>
            <h1>🤖 오늘의 AI 기술 및 시장 동향 보고서</h1>
        </header>
        <main>
            {report_body}
        </main>
        <footer>
            <p style="text-align: center; color: #999;">© 2026 AI News Automation System</p>
        </footer>
    </body>
    </html>
    """
    with open("index.html", "w", encoding="utf-8") as f:
        f.write(html_template)

if __name__ == "__main__":
    driver = setup_driver()
    raw_news = []

    try:
        for cat, url in CATEGORIES.items():
            raw_news.extend(collect_section_news(driver, cat, url))
        
        ai_news = filter_ai_keywords(raw_news)
        final_list = deduplicate_articles(ai_news, threshold=0.2)
        
        report_body = ""
        df_final = pd.DataFrame(final_list, columns=['분류', '제목', '시간', '링크'])
        
        for category in CATEGORIES.keys():
            category_articles = df_final[df_final['분류'] == category].values.tolist()
            report_body += analyze_category_with_gemini(category, category_articles)
        
        generate_html_report(report_body)
        print("\n✅ index.html 생성 완료")

    finally:
        driver.quit()
