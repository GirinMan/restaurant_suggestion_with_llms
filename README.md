# restaurant_suggestion_with_llms
![image](https://github.com/liner-engineering/liner-pdf-chat-tutorial/assets/44901828/d5026b33-9965-4724-9b45-99bfe88ed5fe)
자연어로 된 입력을 받아 맛집 추천을 해주는 프로그램입니다.

## 실행 방법
#### API key 입력
- 네이버 검색 API Client ID/Secret, OpenAI API key 등을 utils 폴더 내부에 있는 소스 코드 파일에 입력
#### Command parser model API 서버 실행
- ```utils/cmd_parser/run_server.sh``` 실행
#### Streamlit app 실행
- ```streamlit run demo.py```

## 프로젝트 소개
- 대규모 언어 모델을 활용한 에이전트를 기반으로, 사람이 단계적으로 작업을 수행하는 과정을 모방하도록 하는 Langchain과 OpenAI 모델 활용
- 좋은 식사 장소를 찾기 위한 과정을 정의하고, 이를 AI agent가 수행하도록 개발 
    -> Norman’s 7-stage action model 활용
- 유저가 직접 검색하여 운영 시간, 주소 등의 추가 정보를 찾아야 하는 맛집 추천 서비스들과 차별화

## 맛집 탐색 과정의 7-stage action model
1. Forming the goal: 식당을 찾기 위한 명령 입력
2. Forming the intention: 현재 시간, 지역, 요구사항을 명령에서 추출
3. Specifying the action: 추출한 정보를 바탕으로 검색 사이트 및 검색어 결정
4. Executing the action: 결정된 키워드로 인터넷에서 검색한 결과 크롤링
5. Perceiving the system state: 검색 결과를 분석하여 현재 상황에 맞는 후보군 압축
6. Interpreting the system state: 현재 상황에 맞게 선택된 식당의 메뉴, 운영 시간 등 확인
7. Evaluating the outcome: 사용자에게 최종 선택된 식당/카페의 정보를 표시

## 실행 과정
1. 프로그램을 실행하고 자연어로 맛집을 찾는 것과 관련된 목표를 지정해줍니다.
2. LLM 에이전트가 현재 시간과 위치, 기존 선호도 등을 기반으로 검색 키워드 등 구체적인 목표를 지정합니다.
3. 네이버 장소 검색 API 등을 활용하여 신뢰할 만한 주요 맛집 블로그나 카페 등 커뮤니티에서 정보를 찾아온 뒤, 검색 결과를 분석하여 현재 상황에 적절한 후보군들을 압축합니다.
4. 결정된 후보군에서 랜덤하게 최종 후보군을 선택합니다. 맛집 리뷰 사이트 조회수나 평점이 높은 장소는 선택될 확률이 높아집니다.
5. 최종 선택된 식당의 정보와, 리뷰 내용 자동 요약 결과를 사용자에게 출력합니다.