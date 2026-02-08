import csv
import time
import re
import os
import json
import pandas as pd
from google import genai  # 최신 SDK 사용
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- 1. 환경 설정 및 카테고리 --- (기존 유지)
CATEGORIES = {
    "모바일": "https://news.naver.com/breakingnews/section/105/731",
    "인터넷 & SNS": "https://news.naver.com/breakingnews/section/105/226",
    "통신 & 뉴미디어": "https://news.naver.com/breakingnews/section/105/227",
    "IT 일반": "https://news.naver.com/breakingnews/section/105/230",
    "컴퓨터": "https://news.naver.com/breakingnews/section/105/283",
    "과학 일반": "https://news.naver.com/breakingnews/section/105/228"
}

client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

def setup_driver():
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
    return webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

# --- 2. 수집 및 정제 엔진 (기존 유지) ---
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

def deduplicate_articles(data_list, threshold=0.4):
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

# --- 3. Gemini 지능형 분석 및 카드뉴스 가공 엔진 (로직 전면 수정됨) ---
def analyze_category_with_gemini(category_name, articles):
    if not articles:
        return None

    print(f"🔎 [{category_name}] 심층 리서치 시작 (총 {len(articles)}건의 기사 분석)...")

    # [수정됨] 1. 개별 기사 순차 검색 및 정보 축적 (정보 누락 방지)
    combined_research_data = ""
    
    for idx, article in enumerate(articles):
        title = article[1]
        print(f"   └ ({idx+1}/{len(articles)}) 검색 수행 중: {title[:20]}...")
        
        # 각 기사별 검색 수행을 위한 미니 프롬프트
        mini_prompt = f"""
        다음 뉴스 기사 제목에 대해 Google 검색을 수행하여 기사 내용에 대한 핵심 내용
        (기사가 말하고자 하는 가장 핵심적이고 중요한 사건, 해당 사건에 대한 해석을 뒷받침하기 위해 기사에서 담은 근거, 해당 사건과 관련된 변화 등)를
        누가(Who), 언제(When), 어디서(Where), 무엇을(What), 어떻게(How), 왜(Why) 중 명시되지 않은 정보를 제외하더라도 최대한 준수해서 핵심 내용을 위주로 3줄 내외로 요약해줘.
        기사 제목: {title}
        """
        
        try:
            # 개별 기사 검색 (Google Search 도구 사용)
            mini_response = client.models.generate_content(
                model='gemini-3-flash-preview', 
                contents=mini_prompt,
                config={'tools': [{'google_search': {}}]}
            )
            combined_research_data += f"\n[기사 {idx+1} 요약: {title}]\n{mini_response.text}\n" + "-"*30
            time.sleep(1) # API 호출 안정성을 위한 짧은 대기
            
        except Exception as e:
            print(f"      ⚠️ 검색 에러 (Skip): {e}")
            combined_research_data += f"\n[기사 {idx+1}: {title}]\n(검색 실패로 인한 제목만 참조)\n"

    # [수정됨] 2. 축적된 정보를 바탕으로 최종 통합 분석
    article_list_str = "\n".join([f"- {a[1]}" for a in articles])
    links_html = "".join([f"<li><a href='{a[3]}' target='_blank'>{a[1]}</a></li>" for a in articles])

    # 최종 분석 프롬프트: 'combined_research_data'를 기반으로 작성하도록 지시
    prompt = f"""
    당신은 IT 전문 데이터 전략가입니다. 
    아래 [수집된 리서치 데이터]를 바탕으로 '{category_name}' 분야의 카드뉴스 제작을 위한 최종 요약본을 만드세요.
    
    [수집된 리서치 데이터]
    {combined_research_data}

    [기사 원문 제목 리스트]
    {article_list_str}

    [작성 지침]
    1. 반드시 위 [수집된 리서치 데이터]에 포함된 내용만을 사실(Fact)로 간주하여 분석하세요.
    2. 여러 기사에 공통적으로 등장하는 내용은 '핵심 이슈'로 분류하세요.
    3. 카드뉴스에 포함하는 정보는 누가(Who), 언제(When), 무엇을(What), 어떻게(How) 위주로 핵심 정보를 서술할 것.

    [출력 형식: 반드시 아래 JSON 구조 유지]
    {{
      "card_issue": "가장 많이 언급되는 핵심 이슈 요약: 현재 해당 분야의 가장 큰 흐름을 2문장으로 요약",
      "card_products": "신제품/신기능 소식: AI 관련 신제품, 신기능, 서비스 출시 및 예정 소식이 있다면 관련 기업명을 포함하여 최대 3문장으로 요약",
      "card_changes": "사회/제도/시장의 변화: AI로 인한 기존 시스템이나 시장 구조의 구체적인 '변화' 내용을 요약",
      "card_terms": "앞서 카드뉴스에 포함한 it관련 전문 용어들을 정리하여 괄호를 사용해 친절하게 풀어서 설명",
      "image_keyword": "이 뉴스들의 핵심 내용을 가장 잘 표현하는 영어 단어 하나 (예: robot, smartphone, server 등)",
      "raw_analysis": "참고용 분석 데이터"
    }}
    """

    try:
        print(f"🤖 [{category_name}] 취합된 데이터를 바탕으로 최종 카드뉴스 생성 중...")
        
        # 최종 생성: 이미 충분한 정보가 Context에 있으므로 여기서는 Search Tool을 필수는 아니지만, 
        # 혹시 모를 검증을 위해 켜두거나 끄셔도 됩니다. 여기서는 Context 집중을 위해 끄거나, 
        # SDK 특성상 그대로 두되 Context 우선을 지시했으므로 안전합니다.
        response = client.models.generate_content(
            model='gemini-3-flash-preview', 
            contents=prompt,
            config={
                'response_mime_type': 'application/json'
            }
        )

        
        json_match = re.search(r'\{.*\}', response.text, re.DOTALL)
        if not json_match:
            print(f"⚠️ {category_name}: JSON 형식을 찾을 수 없습니다.")
            return None
            
        analysis_data = json.loads(json_match.group())
        
        return {
            "category": category_name,
            "issue": analysis_data['card_issue'],
            "products": analysis_data['card_products'],
            "changes": analysis_data['card_changes'],
            "terms": analysis_data['card_terms'],
            "img_seed": analysis_data.get('image_keyword', category_name),
            "links": links_html
        }
    except Exception as e:
        print(f"❌ {category_name} 최종 분석 에러: {e}")
        return None

# --- 4. 웹 변환 및 카드뉴스 레이아웃 ---
def save_as_card_news(analysis_results):
    """간추려진 분석 결과를 5그리드 카드뉴스 형식으로 저장"""
    
    cards_html = ""
    for data in analysis_results:
        if not data: continue
        
        # [수정] f-string 내부 백슬래시 에러 방지를 위해 외부에서 미리 치환
        formatted_issue = data['issue'].replace('\n', '<br>')
        formatted_products = data['products'].replace('\n', '<br>')
        formatted_changes = data['changes'].replace('\n', '<br>')
        formatted_terms = data['terms'].replace('\n', '<br>')
        
        cards_html += f"""
        <div class="category-row">
            <h2 class="category-title">📂 {data['category']} (Hot Topic)</h2>
            <div class="grid-container">
                <div class="card">
                    <div class="card-tag">Core Issue</div>
                    <h3>핵심 이슈</h3>
                    <div class="card-content">{formatted_issue}</div>
                </div>
                <div class="card">
                    <div class="card-tag">New Release</div>
                    <h3>신제품/기능</h3>
                    <div class="card-content">{formatted_products}</div>
                </div>
                <div class="card">
                    <div class="card-tag">Market Change</div>
                    <h3>시장 변화</h3>
                    <div class="card-content">{formatted_changes}</div>
                </div>
                <div class="card">
                    <div class="card-tag">Tech Terms</div>
                    <h3>용어 설명</h3>
                    <div class="card-content">{formatted_terms}</div>
                </div>
                <div class="card links-card">
                    <div class="card-tag">References</div>
                    <div class="links-header">🔗 주요 기사 원문</div>
                    <ul class="links-list">
                        {data['links']}
                    </ul>
                </div>
            </div>
        </div>
        """

    # HTML 템플릿 부분은 동일 (변수 처리 방식만 유지)
    current_time = time.strftime('%Y-%m-%d %H:%M:%S')
    html_template = f"""
    <!DOCTYPE html>
    <html lang="ko">
    <head>
        <meta charset="UTF-8">
        <title>Daily AI Card News</title>
        <style>
            :root {{
                --bg-color: #f8f9fa;
                --card-bg: #ffffff;
                --primary-color: #2d3436;
                --accent-color: #0984e3;
            }}
            body {{ font-family: 'Pretendard', sans-serif; background: var(--bg-color); margin: 0; padding: 20px; }}
            .category-row {{ margin-bottom: 50px; overflow-x: auto; }}
            .category-title {{ border-left: 5px solid var(--accent-color); padding-left: 15px; margin-bottom: 20px; color: var(--primary-color); }}
            .grid-container {{ display: flex; gap: 20px; padding-bottom: 15px; min-width: min-content; }}
            .card {{ background: var(--card-bg); border-radius: 12px; width: 300px; flex-shrink: 0; box-shadow: 0 4px 15px rgba(0,0,0,0.08); padding: 15px; display: flex; flex-direction: column; }}
            .card-tag {{ font-size: 11px; font-weight: bold; color: var(--accent-color); text-transform: uppercase; margin-bottom: 8px; }}
            .card h3 {{ font-size: 18px; margin: 0 0 10px 0; color: #2d3436; }}
            .card-content {{ font-size: 14px; line-height: 1.6; color: #636e72; flex-grow: 1; }}
            .links-card {{ background: #2d3436; color: white; }}
            .links-header {{ font-weight: bold; margin-bottom: 15px; border-bottom: 1px solid #444; padding-bottom: 10px; }}
            .links-list {{ padding-left: 20px; font-size: 13px; color: #dfe6e9; line-height: 1.8; }}
            .links-list a {{ color: #74b9ff; text-decoration: none; }}
        </style>
    </head>
    <body>
        <h1 style="text-align:center; margin-bottom:40px;">🤖 Daily AI 카드뉴스 리포트 (w/ Deep Research)</h1>
        {cards_html}
        <p style="text-align:center; color:gray; margin-top:50px;">Last Updated: {current_time}</p>
    </body>
    </html>
    """
    with open("index.html", "w", encoding="utf-8") as f:
        f.write(html_template)

        
# --- 5. 메인 실행 프로세스 (기존 유지) ---
if __name__ == "__main__":
    driver = setup_driver()
    raw_news = []

    try:
        for cat, url in CATEGORIES.items():
            raw_news.extend(collect_section_news(driver, cat, url))
        
        print(f"\n--- 1단계: 수집 완료 ({len(raw_news)}건) ---")

        ai_news = filter_ai_keywords(raw_news)
        final_list = deduplicate_articles(ai_news, threshold=0.4)
        print(f"✨ 필터링 결과: 수집({len(raw_news)}) -> AI추출({len(ai_news)}) -> 중복제거({len(final_list)})")

        analysis_results = []
        df_final = pd.DataFrame(final_list, columns=['분류', '제목', '시간', '링크'])
        
        for category in CATEGORIES.keys():
            category_articles = df_final[df_final['분류'] == category].values.tolist()
            
            if category_articles:
                res = analyze_category_with_gemini(category, category_articles)
                if res: analysis_results.append(res)
        
        save_as_card_news(analysis_results)
        pd.DataFrame(final_list, columns=['분류','제목','시간','링크']).to_csv("naver_today_news.csv", index=False, encoding='utf-8-sig')
        
        print("\n✅ 모든 작업 완료: index.html 생성됨")

    finally:
        driver.quit()
