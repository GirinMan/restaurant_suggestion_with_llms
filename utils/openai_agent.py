from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain import PromptTemplate

llm = ChatOpenAI(
    openai_api_key="OPENAI_API_KEY",
    temperature=0.5,
    model_name="gpt-3.5-turbo-0301",
    max_tokens=256,
)

search_query_template = """
입력된 지역과 요구사항을 고려하여 적절한 도구를 사용해 맛집 또는 카페를 검색하기 위한 한국어 검색어를 찾아내세요.

아래 형식을 사용하여 대답하세요.

<입출력 형식>
지역: 없음 또는 검색 위치
요구사항: 없음 또는 검색 키워드에 반영할 사항
도구: 망고플레이트(입력된 "요구사항"이 "없음"이거나 "카페", "술집" 처럼 간단한 경우), 네이버지도(입력된 "요구사항"이 2개 이상의 단어로 구성된 경우) 중 1개
검색어: 지역 정보를 포함하는 3개 이하의 단어로 구성된 서로 다른 검색어 5개

이제부터 입출력 형식을 지키며 답변하세요.
생성된 5개의 검색어는 한 줄에 쉼표(,)로 구분하여 출력하세요.
지역: {loc}
요구사항: {pref}
"""

search_query = PromptTemplate(
    input_variables=["loc", "pref"],
    template=search_query_template,
)

search_query_chain = LLMChain(llm=llm, prompt=search_query)

def get_search_query(loc, pref):
    result = search_query_chain.run(loc=loc, pref=pref).strip().split('\n')
    tool_idx, query_idx = 0, 0
    for idx in range(len(result)):
        if "도구:" in result[idx]:
            tool_idx = idx
        if "검색어:" in result[idx]:
            query_idx = idx
    tool = result[tool_idx].split(":")[1].strip()
    query = [text.strip() for text in result[query_idx].split(":")[1].strip().split(",")]
    return {"tool":tool, "query":query}

menu_remove_template = """
아래 <요청>의 목적을 [식사, 음주, 카페] 중 하나로 분류하세요.

출력 형식은 아래와 같습니다:
목적: [식사, 음주, 카페] 중 1개

<요청>
{instruction}
"""

menu_remove = PromptTemplate(
    input_variables=["instruction"],
    template=menu_remove_template,
)

menu_remove_chain = LLMChain(llm=llm, prompt=menu_remove)

def get_order_type(instruction):
    try:
        result = menu_remove_chain.run(instruction=instruction).split(":")[1].strip()

        type_to_targets = {
            "식사": [
                "전통 주점 / 포차",
                "베이커리",
                "칵테일 / 와인",
                "일반 주점",
                "이자카야 / 오뎅 / 꼬치",
                "카페 / 디저트",
            ],
            "음주": [
                "베이커리",
                "브런치 / 버거 / 샌드위치",
                "카페 / 디저트",
            ],
            "카페": [
                "전통 주점 / 포차",
                "패밀리 레스토랑",
                "인도 음식",
                "기타 양식",
                "세계음식 기타",
                "프랑스 음식",
                "탕 / 찌개 / 전골",
                "다국적 퓨전",
                "기타 일식",
                "퓨전 중식",
                "칵테일 / 와인",
                "기타 중식",
                "일반 주점",
                "퓨전 한식",
                "정통 중식 / 일반 중식",
                "이자카야 / 오뎅 / 꼬치",
                "남미 음식",
                "태국 음식",
                "돈부리 / 일본 카레 / 벤토",
                "라멘 / 소바 / 우동",
                "다국적 아시아 음식",
                "이탈리안",
                "퓨전 양식",
                "닭 / 오리 요리",
                "퓨전 일식",
                "기타 한식",
                "스테이크 / 바베큐",
                "뷔페",
                "정통 일식 / 일반 일식",
                "시푸드 요리",
                "고기 요리",
                "딤섬 / 만두",
                "국수 / 면 요리",
                "베트남 음식",
                "회 / 스시",
                "까스 요리",
                "철판 요리",
                "치킨 / 호프 / 펍",
                "해산물 요리",
                "한정식 / 백반 / 정통 한식",
            ],
        }

        targets = type_to_targets[result]
        return {"type":result, "targets":targets}
    except:
        return {"type":"식사", "targets":[]}

review_summary_template = """
아래 식당 리뷰 내용을 세 줄로 요약한 내용을 Markdown 형식으로 출력하세요.

{review}
""".strip()

review_summary = PromptTemplate(
    input_variables=["review"],
    template=review_summary_template,
)

review_summary_chain = LLMChain(llm=llm, prompt=review_summary)

def get_review(review):
    result = review_summary_chain.run(review=review)
    return result


if __name__== "__main__":
    #print(get_search_query("인덕원", "고기"))

    instruction = "한양대 점심 먹을 곳 추천해주세요"

    #print(get_order_type(instruction))

    review = """
신승수
489
1
개안음
2023-04-26
괜찮다
ustar
275
9
겉면이 바삭하고 안에 초코가 아주 고급스러운 맛이었던 기억.
사실 가게에 앉아서 먹을 수 있는곳이 있을 줄 알고 방문했는데 포장해 갈 수 밖에 없었다. 계획에 없던 가게앞 길빵(?)을 할 수 밖에 없었지만 빵맛 만큼은 좋았던 기억이 있다
2022-08-17
맛있다
seul
1265
352
정말 애정하는 폴앤폴리나ㅠㅠ
광화문하면 바로 생각날 정도로 이집 빵을 좋아한다.
담백한 빵 중 가장 취향에 잘 맞는 곳임.
뭘 골라도 다 맛있는데 브레첼류가 베스트다.
부스러기가 없는 빵들이라 길빵하기도 좋음.
2022-07-09
맛있다
Capriccio06
1693
192
화이트치아바타랑 브레첼이 맛있는 집. 식사용으로 소화 잘되고 질리지 않는 맛의 빵들이 주메뉴인데, 다른 빵들도 전반적으로 다 괜찮지만 두 메뉴를 제일 좋아한다. 화이트치아바타는 냉동했다 해동해도 맛이 유지되는 편이고, 샌드위치 메뉴들로도 평이 좋다. 요즘은 식사용 빵 하는 집들도 많아졌고 잘하는 집들도 늘어서 예전 같은 느낌은 아니지만, 그래도 화이트치아바타는 아직까지는 여기가 제일 좋다.

연희동 본점 자주다니다가 광화문점 생긴 이후로는 주로 여기로 가는데 최근 매장이 바뀌었다. 무인점포를 준비하는 느낌으로 매장 구조나 입구가 변경되었는데 운영 방식이 바뀌는지는 모르겠다. 이 날은 직원 한분이 계산을 도와주셨다. 갓 구운 빵을 인심 후하게 시식하게 도와주던게 인기 요인 중 하나였는데 계속 매장이 단촐해져서 코로나 끝나면 예전처럼 운영될지 모르겠다.
2022-06-11
맛있다
MAXIMA
610
160
올리브빵은 여전히 쫄깃하고 맛있다. 올리브유를 뿌려서 먹으면 향긋한게 아주 별미.
버터 브레첼에는 생각보다 버터의 양이 섭섭하게 들어있었다. 요즘 하도 잠봉뵈르 샌드위치 비주얼에 익숙해져서 그런가.. 브레첼을 열어봤더니 뚝,뚝, 끊겨있던 버터가 어찌나 서운하던지..ㅋㅋㅋ어쩔 수 없이 집에 있던 라꽁비에뜨를 넣어 보충해 먹었다.
치아바타는 반 갈라서 에담치즈를 넣어 먹었더니 딱 좋았다.
빵순이로서 동네에 이런 빵집이 있다는 건 그저 큰 행운일 따름이다ㅎㅎ
2022-06-05
맛있다
""".strip()

    print(get_review(review))