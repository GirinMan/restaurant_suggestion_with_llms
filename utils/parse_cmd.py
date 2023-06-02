import requests
import json


def parse_cmd(text:str, adapter_name="latest"):
    # Set the API endpoint URL
    url = "http://localhost:9022/generate"
    prompt = f"입력된 명령에서 시간, 지역, 요구사항을 추출하세요.\n정보를 찾을 수 없을 경우 '없음'으로 표시하세요.\n명령: {text}\n"

    # Create a dictionary containing the file data
    file_data = json.dumps({"text": prompt, "adapter_name":adapter_name})

    # Send a POST request to the API with the file data
    response = requests.post(url, data=file_data)

    # Check if the request was successful (status code 200)
    if response.ok:
        result =  response.json()['text'].strip().split('\n')
        print(result)
        try:
            time = result[-3].split("시간:")[1].strip()
            loc = result[-2].split("지역:")[1].strip()
            pref = result[-1].split("요구사항:")[1].strip()
        except:
            time = loc = pref = "ERROR"
        return {"cmd":text, "time":time, "loc":loc, "pref":pref}
        

    else:
        # If the request failed, print the error message returned by the API
        error_msg = response.json()["detail"]
        print(f"API error: {error_msg}")
        return None

if __name__ == "__main__":

    text = "건대입구 쪽에 괜찮은 술집이나 고깃집 있으면 추천 좀 해주세요~"

    response = parse_cmd(text)
    print(response)


    adapters = [
        "step-15",
        "step-30",
        "step-45",
        "step-60",
        "step-75",
        "step-90",
        "step-105",
        "step-120",
        "step-135",
        "step-150",
    ]

    for adapter in adapters:
        response = parse_cmd(text, adapter)
        print("adapter name:", adapter)
        print(response)
