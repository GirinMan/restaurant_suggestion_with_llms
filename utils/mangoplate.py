import sys # 시스템
import os  # 시스템

# 데이터 다루기
import pandas as pd
import numpy as np

# selenium 크롤링
from selenium import webdriver  
from selenium.webdriver import ActionChains as AC
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By

# 크롬 드라이버
import chromedriver_autoinstaller

# beautifulsoup 크롤링
import requests
from bs4 import BeautifulSoup

# lxml 크롤링
import lxml.html

# 시간 조절
import time

# 시간 측정
from tqdm import notebook

# 정규표현식
import re

# 경고 무시
import warnings
warnings.filterwarnings('ignore')

# 키워드 URL 검색용으로 변환
from urllib import parse

# 크롬창 띄우기
def get_driver():
    chrome_path = chromedriver_autoinstaller.install()
    driver = webdriver.Chrome(chrome_path)
    return driver

# 키워드에 해당하는 음식점 리스트 탐색
def getFoodList(driver, keyword, page=1, row=10, col=2):
    # 전체 담을 그릇
    total_dict = {}
    k = 0
    
    keyword = parse.quote(keyword)
    
    for p in range(1, page+1):
        # 키워드가 입력된 망고플레이트 사이트에 들어가기
        try:
            driver.get("https://www.mangoplate.com/search/{0}?keyword={0}&page={1}".format(keyword, p))
            driver.execute_script("window.scrollTo(0, 2000)")
        except:
            break
        entry_count = 0

        for i in range(row):
            for j in range(col):
                # 동영상 1개 담을 그릇
                sub_dict = {}
                
                # 음식점 클릭
                title_element = f'body > main > article > div.column-wrapper > div > div > section > div.search-list-restaurants-inner-wrap > ul > li:nth-child({i+1}) > div:nth-child({j+1}) > figure > figcaption > div > a > h2'    
                point_element = f'body > main > article > div.column-wrapper > div > div > section > div.search-list-restaurants-inner-wrap > ul > li:nth-child({i+1}) > div:nth-child({j+1}) > figure > figcaption > div > strong'
                view_element = f'body > main > article > div.column-wrapper > div > div > section > div.search-list-restaurants-inner-wrap > ul > li:nth-child({i+1}) > div:nth-child({j+1}) > figure > figcaption > div > p.etc_info > span.view_count'
                review_element = f'body > main > article > div.column-wrapper > div > div > section > div.search-list-restaurants-inner-wrap > ul > li:nth-child({i+1}) > div:nth-child({j+1}) > figure > figcaption > div > p.etc_info > span.review_count'
                etc_element = f'body > main > article > div.column-wrapper > div > div > section > div.search-list-restaurants-inner-wrap > ul > li:nth-child({i+1}) > div:nth-child({j+1}) > figure > figcaption > div > p.etc'
                url_element = f'body > main > article > div.column-wrapper > div > div > section > div.search-list-restaurants-inner-wrap > ul > li:nth-child({i+1}) > div:nth-child({j+1}) > figure > figcaption > div > a'
                
                name_with_selectors = {
                    "title": title_element,
                    "point": point_element,
                    "view": view_element,
                    "review": review_element,
                    "etc": etc_element,
                    "url": url_element,
                }
                try:
                    for name, selector in name_with_selectors.items():
                        element = driver.find_element(By.CSS_SELECTOR, selector)
                        if name == "url":
                            sub_dict[name] = element.get_attribute('href')
                        elif name == "etc":
                            desc = element.text
                            loc, menu = desc.split(sep=" - ")
                            sub_dict["loc"] = loc
                            sub_dict["menu"] = menu
                        else:
                            sub_dict[name] = element.text
                    entry_count += 1
                except:
                    continue

                # total_dict에 담기
                total_dict[k] = sub_dict
                k += 1
        if entry_count == 0:
            break    

    df = pd.DataFrame.from_dict(total_dict, orient='index')
    return df

# 음식점 1개에서 정보 크롤링 함수
def getFoodInfo(driver, url):    
    # 키워드가 입력된 망고플레이트 사이트에 들어가기
    driver.get(url)
    cur_url = driver.current_url
    sub_dict = dict()

    # 가게 이름 크롤링
    element = 'body > main > article > div.column-wrapper > div.column-contents > div > section.restaurant-detail > header > div.restaurant_title_wrap > span > h1'
    title_raw = driver.find_element(By.CSS_SELECTOR, element)
    title = title_raw.text

    # 가게 전체 평점 점수 크롤링
    try:
        element = 'body > main > article > div.column-wrapper > div.column-contents > div > section.restaurant-detail > header > div.restaurant_title_wrap > span > strong > span'
        total_raw = driver.find_element(By.CSS_SELECTOR, element)
        total = float(total_raw.text)
    except:
        total = 0.

    # 조회수 크롤링
    try:
        element = 'body > main > article > div.column-wrapper > div.column-contents > div > section.restaurant-detail > header > div.status.branch_none > span.cnt.hit'
        view_raw = driver.find_element(By.CSS_SELECTOR, element)
        view = view_raw.text
    except:
        element = 'body > main > article > div.column-wrapper > div.column-contents > div > section.restaurant-detail > header > div.status > span.cnt.hit'
        view_raw = driver.find_element(By.CSS_SELECTOR, element)
        view = view_raw.text

    # 리뷰 개수 크롤링
    try:
        element = 'body > main > article > div.column-wrapper > div.column-contents > div > section.restaurant-detail > header > div.status.branch_none > span.cnt.review'
        review_raw = driver.find_element(By.CSS_SELECTOR, element)
        num_review = review_raw.text

    except:
        element = 'body > main > article > div.column-wrapper > div.column-contents > div > section.restaurant-detail > header > div.status > span.cnt.review'
        review_raw = driver.find_element(By.CSS_SELECTOR, element)
        num_review = review_raw.text

    # 별표 개수 크롤링
    try:
        element = 'body > main > article > div.column-wrapper > div.column-contents > div > section.restaurant-detail > header > div.status.branch_none > span.cnt.favorite'
        star_raw = driver.find_element(By.CSS_SELECTOR, element)
        num_star = star_raw.text

    except:
        element = 'body > main > article > div.column-wrapper > div.column-contents > div > section.restaurant-detail > header > div.status > span.cnt.favorite'
        star_raw = driver.find_element(By.CSS_SELECTOR, element)
        num_star = star_raw.text
    
    # 가격대 크롤링
    price = ""
    for idx in range(10):
        try:
            tr_head = f'body > main > article > div.column-wrapper > div.column-contents > div > section.restaurant-detail > table > tbody > tr:nth-child({idx+1}) > th'
            head_raw = driver.find_element(By.CSS_SELECTOR, tr_head)
            head = head_raw.text
            if '가격대' == head.strip():
                element1 = f'body > main > article > div.column-wrapper > div.column-contents > div > section.restaurant-detail > table > tbody > tr:nth-child({idx+1}) > td'
                type_raw = driver.find_element(By.CSS_SELECTOR, element1)
                type_raw = type_raw.text
                price = type_raw
                break
        except:
            pass

    # 영업 시간 크롤링
    open_hour = ""
    breaktime = ""
    closed_day = ""
    for idx in range(10):
        try:
            tr_head = f'body > main > article > div.column-wrapper > div.column-contents > div > section.restaurant-detail > table > tbody > tr:nth-child({idx+1}) > th'
            head_raw = driver.find_element(By.CSS_SELECTOR, tr_head)
            head = head_raw.text
            if '영업시간' == head.strip():
                element1 = f'body > main > article > div.column-wrapper > div.column-contents > div > section.restaurant-detail > table > tbody > tr:nth-child({idx+1}) > td'
                type_raw = driver.find_element(By.CSS_SELECTOR, element1)
                type_raw = type_raw.text
                open_hour = type_raw
            if '쉬는시간' == head.strip():
                element1 = f'body > main > article > div.column-wrapper > div.column-contents > div > section.restaurant-detail > table > tbody > tr:nth-child({idx+1}) > td'
                type_raw = driver.find_element(By.CSS_SELECTOR, element1)
                type_raw = type_raw.text
                breaktime = type_raw
            if '휴일' == head.strip():
                element1 = f'body > main > article > div.column-wrapper > div.column-contents > div > section.restaurant-detail > table > tbody > tr:nth-child({idx+1}) > td'
                type_raw = driver.find_element(By.CSS_SELECTOR, element1)
                type_raw = type_raw.text
                closed_day = type_raw
        except:
            pass
        
    # 음식 종류 및 메뉴 크롤링
    for idx in range(10):
        try:
            tr_head = f'body > main > article > div.column-wrapper > div.column-contents > div > section.restaurant-detail > table > tbody > tr:nth-child({idx+1}) > th'
            head_raw = driver.find_element(By.CSS_SELECTOR, tr_head)
            head = head_raw.text
            if '음식 종류' == head.strip():
                element1 = f'body > main > article > div.column-wrapper > div.column-contents > div > section.restaurant-detail > table > tbody > tr:nth-child({idx+1}) > td'
                type_raw = driver.find_element(By.CSS_SELECTOR, element1)
                type_raw = type_raw.text
                food_type = [type_raw, []]
                break
        except:
            pass
    
    is_menu_available = False
    for idx in range(10):
        try:
            tr_head = f'body > main > article > div.column-wrapper > div.column-contents > div > section.restaurant-detail > table > tbody > tr:nth-child({idx+1}) > th'
            head_raw = driver.find_element(By.CSS_SELECTOR, tr_head)
            head = head_raw.text
            if '메뉴' in head:
                is_menu_available = True
                break
        except:
            pass

    if is_menu_available:
        try:
            element2 = f'body > main > article > div.column-wrapper > div.column-contents > div > section.restaurant-detail > table > tbody > tr:nth-child({idx+1}) > td > ul'
            menu_raw = driver.find_element(By.CSS_SELECTOR, element2)
            food_menu = menu_raw.text
            food_menu = food_menu.split('\n')
            food_menu = [(food_menu[2*i], food_menu[2*i+1]) for i in range(int(len(food_menu)/2))]
            food_type = [type_raw, food_menu]
        except:
            pass

    review_raw = driver.find_element(By.CSS_SELECTOR, "body > main > article > div.column-wrapper > div.column-contents > div > section.RestaurantReviewList > ul").text
    try:
        thumnail_element = driver.find_element(By.CSS_SELECTOR, "body > main > article > aside.restaurant-photos > div > div.owl-wrapper-outer > div > div:nth-child(1) > figure > figure > img")
        thumnail_url = thumnail_element.get_attribute('src')
    except:
        thumnail_url = "https://web.yonsei.ac.kr/_ezaid/board/_skin/albumRecent/1/no_image.gif"
    addr_element = driver.find_element(By.CSS_SELECTOR, 'body > main > article > div.column-wrapper > div.column-contents > div > section.restaurant-detail > table > tbody > tr:nth-child(1) > td > span.Restaurant__InfoAddress--Text')
    addr = addr_element.text

    # sub_dict에 담기
    sub_dict['title'] = title
    sub_dict['Point'] = total
    sub_dict['View'] = view
    sub_dict['Review'] = num_review
    sub_dict['Star'] = num_star
    sub_dict["Open_Hour"] = open_hour
    sub_dict["Breaktime"] = breaktime
    sub_dict["Closed_Day"] = closed_day
    sub_dict["Price"] = price
    sub_dict['Type'] = food_type[0]
    sub_dict["Menu"] = food_type[1]
    sub_dict['URL'] = cur_url
    sub_dict['Review_Text'] = review_raw
    sub_dict['Thumnail'] = thumnail_url.replace("512:512", "256:256")
    sub_dict['Address'] = addr
            
    return sub_dict