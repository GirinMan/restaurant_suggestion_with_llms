import sys
import requests
import json
import pandas as pd

# 키워드 URL 검색용으로 변환
from urllib import parse

client_id = "CLIENT_ID"
client_secret = "CLIENT_SECRET"


headers = {
    "X-Naver-Client-Id": client_id,
    "X-Naver-Client-Secret": client_secret,
}

def get_place_info(query, display=5, sort="random"):
    url = f"https://openapi.naver.com/v1/search/local.json?query={query}&display={display}&start=1&sort={sort}"
    response = requests.get(url=url, headers=headers)
    rescode = response.status_code
    if(rescode == 200):
        return pd.DataFrame(json.loads(response.text)['items'])
    else:
        print(f"Error({rescode}) : " + response.text)
        return None
    
def get_webpage_info(query, display=100):
    url = f"https://openapi.naver.com/v1/search/webkr.json?query={query}&display={display}&start=1"
    response = requests.get(url=url, headers=headers)
    rescode = response.status_code
    if(rescode == 200):
        return pd.DataFrame(json.loads(response.text)['items'])
    else:
        print(f"Error({rescode}) : " + response.text)
        return None

if __name__ == '__main__':
    result = get_webpage_info("한양대 맛집")
    
    df = pd.DataFrame(result)
    print(df)
