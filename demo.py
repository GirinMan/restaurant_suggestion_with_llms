import streamlit as st
from streamlit.components.v1 import html
from streamlit_js_eval import streamlit_js_eval, get_user_agent, get_geolocation
from audio_recorder_streamlit import audio_recorder
import json
import time, datetime
import pandas as pd
import numpy as np
import random
from utils.time_and_locations import reverse_geocode, get_geocode, map_js
from utils.parse_cmd import parse_cmd
from utils.clova_speech import get_clova_stt
from utils.openai_agent import get_search_query, get_order_type, get_review
from utils.mangoplate import get_driver, getFoodInfo, getFoodList
from utils.naver_search import get_place_info

st.set_page_config(layout="wide")


def main(driver):
    """ NLP Based App with Streamlit """

    # 현재 날짜 계산
    KST = datetime.timezone(datetime.timedelta(hours=9))
    timestamp = datetime.datetime.fromtimestamp(time.time(), tz=KST)
    year, month, day = str(timestamp).split()[0].split(sep='-')
    cur_time = int(str(timestamp).split()[1].split(":")[0])
    if cur_time < 11:
         cur_time = "아침"
    elif cur_time < 17:
         cur_time = "점심"
    else:
         cur_time = "저녁"
    weekday = ["월", "화", "수", "목", "금", "토", "일"][timestamp.weekday()]

    # 현재 위치 출력
    try:
        loc = get_geolocation()
        latitude = loc['coords']['latitude']
        longitude = loc['coords']['longitude']
        geocode = f"{longitude},{latitude}"
        admcode = reverse_geocode(geocode)
    except:
        admcode = "서울시", "성동구", "성수동"
    
    # Title
    st.title("LLM 기반 맛집 추천")
    st.text_input(f"**현재 날짜**", f"{year}년 {month}월 {day}일 ({weekday}요일)")
    st.markdown(f"**현재 시간:** {cur_time}")
    cur_loc = st.text_input(f"**현재 위치**", f"{admcode[1]}")

    sample_input = "한양대 점심 먹을 곳 추천해주세요"
    parsed_cmd = None
    
    debug = st.checkbox("중간 과정 표시", value=False)

    with st.expander("명령 입력", expanded=True):
        input_type = st.selectbox("입력 방식", options=["텍스트", "음성"])
        if input_type == "음성":
            col1, col2 = st.columns([1, 3])
            with col1:
                audio_bytes = audio_recorder(
                    text="요청사항 입력",
                    recording_color="#e8b62c",
                    neutral_color="#6aa36f",
                    icon_name="microphone",
                    icon_size="6x",
                )
            if audio_bytes:
                query = get_clova_stt(audio_bytes)["text"]
                if query is not None:
                    col2.subheader(query)
                    parsed_cmd = parse_cmd(query, adapter_name="step-90")
        
        elif input_type == "텍스트":
            query = st.text_input("요청사항 입력", sample_input)
            if st.button("맛집 탐색 시작!", type='primary'):
                with st.spinner("요청사항 분석중"):
                    parsed_cmd = parse_cmd(query, adapter_name="step-60")
                    for time_keword in ["아침", "점심", "저녁"]:
                        if time_keword in query and parsed_cmd["time"] == "없음":
                            parsed_cmd["time"] = time_keword

    final_candidates = False
    spinner_loc = st.empty()
    with st.expander("검색 과정", expanded=debug):                
        if parsed_cmd is not None:
            if parsed_cmd['time'] == "없음":
                parsed_cmd['time'] = cur_time
            if parsed_cmd['loc'] == "없음":
                parsed_cmd["loc"] = cur_loc
            st.subheader("입력 요청 분석 결과")
            st.write(parsed_cmd)
            
            with spinner_loc:
                with st.spinner("검색 방식 및 검색어 생성중"):
                    search_query = get_search_query(loc=parsed_cmd['loc'], pref=parsed_cmd['pref'])
            st.subheader("요청사항 기반 검색 툴 및 검색어")
            st.write(search_query)
            
            food_list = pd.DataFrame()
            search_result = None

            if search_query["tool"] == "망고플레이트":
                with spinner_loc:
                    with st.spinner("망고플레이트 검색 결과 확인중"):
                        for _query in search_query["query"]:
                            food_list = getFoodList(driver, _query, page=3)
                            if not food_list.empty:
                                break
            
            if food_list.empty:
                search_query["tool"] = "네이버지도"
            
            if search_query["tool"] == "네이버지도":
                search_result = pd.DataFrame()
                with spinner_loc:
                    with st.spinner("네이버 지도 검색 결과 확인중"):
                        for _query in search_query["query"]:
                            search_result = pd.concat((search_result, get_place_info(_query)))
                with spinner_loc:
                    with st.spinner("망고플레이트 검색 결과 확인중"):
                        for name in search_result["title"].unique():
                            name = name.replace("<b>", "").replace("</b>", "")
                            food_list = pd.concat((food_list, getFoodList(driver, name, row=1, col=1))).reset_index(drop=True)
                    
                    

            if search_result is not None and not search_result.empty:
                st.subheader("네이버지도 검색 결과")
                st.write(search_result)
            
            if not food_list.empty:
                for i, point in enumerate(food_list["point"]):
                    if point is None or point == "":
                        food_list["point"][i] = 2.5
                    else:
                        food_list["point"][i] = float(point)
                st.subheader("망고플레이트 검색 결과")
                st.write(food_list)

                st.subheader("식당 카테고리 목록")
                menus = food_list.menu.unique()
                st.write(menus)

                with spinner_loc:
                    with st.spinner("부적절한 메뉴 파악 및 제거중"):
                        menu_analysis = get_order_type(instruction=query)
                menu_type = menu_analysis["type"]
                target_to_remove = menu_analysis["targets"]

                st.subheader("입력 기반 카테고리 정제 결과")
                st.markdown(f"**명령 타입:** {menu_type}")
                st.markdown(f"**제거 대상:** {target_to_remove}")

                filtered_idx = []
                for menu in food_list["menu"]:
                    if menu in target_to_remove:
                        filtered_idx.append(False)
                    else:
                        filtered_idx.append(True)
                food_list = food_list[filtered_idx]

                st.subheader("카테고리 정제 결과")
                st.write(food_list)
                

                st.subheader("평점 기반 랜덤 후보군 선택 결과")
                food_list_idx = [i for i in range(len(food_list["point"]))]
                food_list_probs = list(food_list["point"] / sum(food_list["point"]))
                selected_idx = np.random.choice(food_list_idx, size=min(len(food_list["point"]), 3), replace=False, p=food_list_probs)
                food_list_idx_bool = [False for _ in range(len(food_list["point"]))]
                for i in selected_idx:
                    food_list_idx_bool[i] = True
                food_list = food_list[food_list_idx_bool]
                st.write(food_list)
                final_candidates = True

            else:
                with spinner_loc:
                    st.warning("검색 결과를 찾을 수 없습니다.\n다른 명령으로 다시 시도해보세요.")

    if final_candidates:
        columns = st.columns([1] * len(food_list.index))
        for i, col in enumerate(columns):
            idx = food_list.index[i]
            col.subheader(f"후보 {i+1}번: {food_list['title'][idx]}")
            col.markdown(f'[망고플레이트 식당 정보]({food_list["url"][idx]})')
            food_info = getFoodInfo(driver, food_list['url'][idx])

            with col.expander(f"[지도 보기] {food_info['Address']}", expanded=False):
                x, y = get_geocode(food_info['Address'])
                df_geocode = pd.DataFrame([[float(y), float(x)]], columns=['lat', 'lon'])
                st.map(df_geocode)


            col.markdown(f"![image]({food_info['Thumnail']})")
            menu_str = '\n'.join(['    - ' + item[0] + ' - - - - ' + item[1] for item in food_info['Menu']])
            col.markdown(f"""
- **카테고리:** {food_info['Type']}
- **평점:** {food_info['Point']}/5.0
- **영업시간:** {food_info["Open_Hour"]}
- **브레이크타임:** {food_info["Breaktime"]}
- **휴무일:** {food_info["Closed_Day"]}
- **가격대:** {food_info["Price"]}
- **메뉴 정보**
{menu_str}
            """)

            with col:
                with st.spinner(f"{food_info['title']} 리뷰 요약중"):
                    review_summary = get_review(food_info["Review_Text"])
            
            with col.expander("주요 리뷰 요약", expanded=True):
                st.markdown(review_summary)
            
if __name__ == '__main__':
    driver = get_driver()
    driver.minimize_window()
    main(driver)
    driver.close()
