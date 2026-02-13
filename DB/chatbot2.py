import os
from typing import TypedDict, List
from PIL import Image
from dotenv import load_dotenv, find_dotenv
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

# 1. 환경 변수 로드 (상위 폴더의 .env 자동 검색)
load_dotenv(find_dotenv())
API_KEY = os.getenv("API_KEY")

print(f"API_KEY: {API_KEY[:10]}..." if API_KEY else "API_KEY가 없습니다!")

# 2. State 정의
class PipelineState(TypedDict):
    a_image_path: str
    scenario: str
    a_features: str
    hypotheses: str
    final_report: str

# 3. 모델 설정
llm = ChatGoogleGenerativeAI(
    model="gemini-flash-latest",  # 이 모델 이름 사용
    google_api_key=API_KEY
)

# ============================================
# Agent 1: A안 이미지 분석 (사용자 프롬프트 유지)
# ============================================
def agent_1_analyze_a(state: PipelineState) -> PipelineState:
    print("🔍 [Agent 1] A안 이미지 분석 중...")
    
    # 이미지 파일 존재 여부 확인
    if not os.path.exists(state["a_image_path"]):
        error_msg = f"❌ 이미지를 찾을 수 없습니다: {os.path.abspath(state['a_image_path'])}"
        print(error_msg)
        state["a_features"] = error_msg
        return state

    try:
        img = Image.open(state["a_image_path"])
        
        # 프롬프트
        prompt = """광고 이미지 분석 전문가로서 이미지의 구성 요소를 분석하세요.
    - 카피(문구), 비주얼 오브젝트, 레이아웃, 컬러감을 상세히 기술하세요.
    - 현재 이미지에서 사용자의 '클릭'을 방해하는 요소가 무엇인지 추측해 보세요."""
        
        message = HumanMessage(content=[
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": img}
        ])
        
        response = llm.invoke([message])
        state["a_features"] = response.content
    except Exception as e:
        state["a_features"] = f"이미지 분석 중 오류 발생: {e}"
        
    return state

# ============================================
# Agent 2&3: 가설 수립 및 선택 (개선됨)
# ============================================
def agent_2_3_logic(state: PipelineState) -> PipelineState:
    print("💡 [Agent 2&3] 가설 수립 및 전략 선택 중...")
    
    # 이전 단계 결과가 없으면 중단 방지
    a_info = state.get("a_features", "분석 데이터 없음")
    
    prompt = f"""
[마케팅 시나리오]: {state['scenario']}
[A안 분석 결과]: {a_info}

당신은 데이터 기반 A/B 테스트 전문가입니다. 위 상황을 바탕으로 다음 단계를 수행하세요:

## 1단계: 가설 3가지 수립
클릭률(CTR)을 높이기 위한 실험 가설을 **3가지** 명확히 구분하여 제시하세요.
각 가설은 다음 형식으로 작성:

**[가설 1]: (제목)**
- 문제점: 
- 개선 방향:
- 기대 효과:

**[가설 2]: (제목)**
- 문제점:
- 개선 방향:
- 기대 효과:

**[가설 3]: (제목)**
- 문제점:
- 개선 방향:
- 기대 효과:

## 2단계: 최종 가설 선택 및 근거
위 3가지 가설 중 **가장 타당한 1가지**를 선택하고, 다음 관점에서 구체적인 근거를 제시하세요:
- 왜 이 가설이 CTR 개선에 가장 효과적인가?
- A안 분석 결과와 어떻게 연결되는가?
- 다른 가설 대비 우선순위가 높은 이유는?

**[최종 선택]: 가설 X**
(근거를 3-4문장으로 구체적으로 작성)
"""
    
    response = llm.invoke(prompt)
    state["hypotheses"] = response.content
    return state

# ============================================
# Agent 4: 최종 기획서 작성 (개선됨)
# ============================================
def agent_4_final_report(state: PipelineState) -> PipelineState:
    print("📝 [Agent 4] 최종 B안 기획서 작성 중...")
    
    hypo_info = state.get("hypotheses", "가설 데이터 없음")
    
    prompt = f"""
당신은 시니어 퍼포먼스 마케터입니다. 아래 가설 분석 결과를 바탕으로 최종 리포트를 작성하세요.

[가설 분석 결과]: 
{hypo_info}

**작성 구조:**

# A/B 테스트 가설 분석 및 B안 제작 기획서

## [PART 1] 가설 수립
다음 3가지 가설을 **그대로 재제시**하세요:

**[가설 1]: (제목)**
- 문제점:
- 개선 방향:
- 기대 효과:

**[가설 2]: (제목)**
- 문제점:
- 개선 방향:
- 기대 효과:

**[가설 3]: (제목)**
- 문제점:
- 개선 방향:
- 기대 효과:

---

## [PART 2] 최종 가설 선택 및 근거

**선택된 가설: [가설 X]**

**선택 근거:**
(3-4문장으로 구체적인 근거를 작성)

---

## [PART 3] B안 제작 기획서

### 1. 실험명 및 KPI 목표
- 실험명:
- 배경:
- 핵심 KPI:
- 보조 KPI:

### 2. B안 수정 포인트 (Detailed Changes)
- 구체적인 수정 사항을 표 또는 리스트로 상세히 작성
- 변경 전/후를 명확히 비교

### 3. 디자인 가이드라인 (Design Guidelines)
- 폰트 및 강조점
- 컬러 전략
- 레이아웃 원칙

### 4. 기대 효과 (Expected Impact)
- 정량적/정성적 기대 효과를 구체적으로 서술

---

위 3개 PART를 모두 포함하여 완전한 리포트를 작성해 주세요.
"""
    
    response = llm.invoke(prompt)
    state["final_report"] = response.content
    return state

# ============================================
# 그래프 구축 및 실행
# ============================================
def create_app():
    workflow = StateGraph(PipelineState)
    workflow.add_node("analyze_a", agent_1_analyze_a)
    workflow.add_node("hypothesize", agent_2_3_logic)
    workflow.add_node("report", agent_4_final_report)
    
    workflow.set_entry_point("analyze_a")
    workflow.add_edge("analyze_a", "hypothesize")
    workflow.add_edge("hypothesize", "report")
    workflow.add_edge("report", END)
    return workflow.compile()

if __name__ == "__main__":
    app = create_app()
    
    # 💡 경로 문제 해결: 실행 위치에 상관없이 data 폴더를 찾도록 설정
    # 현재 파일(chatbot.py) 위치를 기준으로 한 단계 위(..)의 data 폴더 이동
    current_dir = os.path.dirname(os.path.abspath(__file__))
    target_img = os.path.join(current_dir, "..", "data", "올리브영.png")

    # 만약 .png가 없으면 .jpg로도 시도 (사용자 업로드 파일 대응)
    if not os.path.exists(target_img):
        target_img = target_img.replace(".png", ".jpg")

    initial_state = {
        "a_image_path": target_img, 
        "scenario": """당신은 '쿠팡' 앱의 마케터입니다. 
        '쿠팡플레이' 설치를 높이는 것이 목표입니다. '와우회원 100% 당첨'이라는 최상단 배너가 있지만 클릭률이 저조합니다."""
    }

    print(f"📍 실행 경로: {os.getcwd()}")
    print(f"📍 이미지 경로: {target_img}")

    result = app.invoke(initial_state)
    
    print("\n" + "="*60)
    print("🧾 [최종 A/B 테스트 기획 리포트]")
    print(result.get("final_report", "결과 생성 실패"))
    print("="*60)