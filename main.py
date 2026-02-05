import csv
import time
import re
import os
import pandas as pd
import google.generativeai as genai
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- 1. 환경 설정 및 Gemini 초기화 ---
CATEGORIES = {
    "모바일": "https://news.naver.com/breakingnews/section/105/731",
    "인터넷 & SNS": "https://news.naver.com/breakingnews/section/105/226",
    "통신 & 뉴미디어": "https://news.naver.com/breakingnews/section/105/227",
    "IT 일반": "https://news.naver.com/breakingnews/section/105/230",
    "컴퓨터": "https://news.naver.com/breakingnews/section/105/283",
    "과학 일반": "https://news.naver.com/breakingnews/section/105/228"
}

genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))

# Google Search Retrieval 도구 활성화
model = genai.GenerativeModel(
    model_name='gemini-1.5-flash',
    tools=[{"google_search_retrieval": {}}]
)

def setup_driver():
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
    return webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

# --- 2. 뉴스 수집 및 필터링 로직 (기존 유지) ---

def filter_ai_keywords(data_list):
    filtered_data = []
    pattern = re.compile(r'ai|인공지능', re.IGNORECASE)
    for item in data_list:
        if pattern.search(item[1]):
            filtered_data.append(item)
    return filtered_data

def deduplicate_articles(data_list, threshold=0.2):
    if not data_list: return []
    df = pd.DataFrame(data_list, columns=['분류', '제목', '시간', '링크'])
    final_indices = []
    for category in df['분류'].unique():
        cat_df = df[df['분류'] == category].copy()
        if len(cat_df) <= 1:
            final_indices.extend(cat_df.index.tolist()); continue
        titles = cat_df['제목'].apply(lambda x: re.sub(r'[^가-힣\s]', '', x)).tolist()
        matrix = TfidfVectorizer().fit_transform(titles)
        sim = cosine_similarity(matrix, matrix)
        keep = [True] * len(cat_df)
        for i in range(len(cat_df)):
            if not keep[i]: continue
            for j in range(i+1, len(cat_df)):
                if sim[i, j] > threshold: keep[j] = False
        final_indices.extend(cat_df.iloc[keep].index.tolist())
    return df.loc[final_indices].values.tolist()

# --- 3. Gemini 지능형 분석 로직 (신규) ---

def analyze_category_with_gemini(category_name, articles):
    """분류별 기사 리스트를 Gemini에게 전달하여 구글 검색 기반 분석 수행"""
    if not articles:
        return f"### {category_name}\n수집된 주요 AI 뉴스가 없습니다.\n"

    # 기사 제목과 링크 리스트화
    article_list_str = "\n".join([f"- {a[1]} ({a[3]})" for a in articles[:10]]) # 카테고리당 최대 10개 분석

    prompt = f"""
    당신은 IT 전문 분석가입니다. 아래 제공된 '{category_name}' 분야의 뉴스 제목들을 구글 검색으로 확인하고 정독한 뒤, 다음 규칙에 따라 리포트를 작성하세요.

    [분석 제외 대상]
    - AI가 기사 내용의 핵심이 아닌 경우
    - 단순히 주가 움직임, 시가총액 등 지나친 경제/금융 중심 뉴스
    - 구체적인 정보 없이 일반적인 인사이트만 다루는 기사 (예: 'AI 공습, 상상력이 무기다' 등)

    [작성 규칙]
    1. 가장 많이 언급되는 핵심 이슈 요약: 현재 해당 분야의 가장 큰 흐름을 2문장으로 요약하고 관련 링크를 제공하세요.
    2. 신제품/신기능 소식: AI 관련 신제품, 신기능, 서비스 출시 및 예정 소식이 있다면 최대 3문장으로 요약하세요.
    3. 사회/제도/시장의 변화: AI로 인한 기존 시스템이나 시장 구조의 구체적인 '변화' 양상을 요약하세요.
    4. **[필수] 전문 용어는 괄호를 사용해 친절하게 풀어서 설명하세요.**

    뉴스 리스트:
    {article_list_str}
    """

    try:
        print(f"🤖 Gemini가 [{category_name}] 분야를 분석 중입니다...")
        response = model.generate_content(prompt)
        return f"## 📌 {category_name} 동향 분석\n{response.text}\n\n"
    except Exception as e:
        return f"## 📌 {category_name} 동향 분석\n분석 중 에러 발생: {e}\n\n"

# --- 4. 메인 실행 프로세스 ---

if __name__ == "__main__":
    driver = setup_driver()
    raw_news = []

    try:
        # 단계 1: 뉴스 수집
        for cat, url in CATEGORIES.items():
            # (기존 collect_section_news 함수 호출부 - 1일전 기사까지 수집)
            # 여기서는 편의상 수집 로직이 작동하여 raw_news에 담겼다고 가정
            pass 
        
        # 단계 2: AI 필터링 및 중복 제거
        ai_news = filter_ai_keywords(raw_news)
        final_list = deduplicate_articles(ai_news, threshold=0.2)
        
        # 단계 3: 분류별 그룹화 및 Gemini 분석
        report_content = ["# 🤖 오늘의 AI 기술 및 시장 동향 보고서\n\n"]
        df_final = pd.DataFrame(final_list, columns=['분류', '제목', '시간', '링크'])
        
        for category in CATEGORIES.keys():
            category_articles = df_final[df_final['분류'] == category].values.tolist()
            analysis = analyze_category_with_gemini(category, category_articles)
            report_content.append(analysis)
        
        # 단계 4: 최종 마크다운 리포트 저장
        with open("AI_Daily_Report.md", "w", encoding="utf-8") as f:
            f.writelines(report_content)
        
        print("\n✨ 분석 리포트 생성이 완료되었습니다: AI_Daily_Report.md")

    finally:
        driver.quit()
