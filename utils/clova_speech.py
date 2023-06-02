import sys
import requests
import json

client_id = "CLIENT_ID"
client_secret = "CLIENT_SECRET"

lang = "Kor" # 언어 코드 ( Kor, Jpn, Eng, Chn )
url = "https://naveropenapi.apigw.ntruss.com/recog/v1/stt?lang=" + lang

headers = {
    "X-NCP-APIGW-API-KEY-ID": client_id,
    "X-NCP-APIGW-API-KEY": client_secret,
    "Content-Type": "application/octet-stream"
}

def get_clova_stt(data):
    response = requests.post(url,  data=data, headers=headers)
    rescode = response.status_code
    if(rescode == 200):
        return json.loads(response.text)
    else:
        print("Error : " + response.text)
        return None

if __name__ == '__main__':
    test_data = open('./test.wav', 'rb')
    result = get_clova_stt(test_data)
    print(type(result))
    print(result)
    print(result["text"])
