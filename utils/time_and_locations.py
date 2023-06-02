import os
import sys
import urllib.request
import json
# 키워드 URL 검색용으로 변환
from urllib import parse

client_id = "CLIENT_ID"
client_secret = "CLIENT_SECRET"

def get_geocode(query: str):
    url = f"https://naveropenapi.apigw.ntruss.com/map-geocode/v2/geocode?query={parse.quote(query)}"
    request = urllib.request.Request(url)
    request.add_header("X-NCP-APIGW-API-KEY-ID", client_id)
    request.add_header("X-NCP-APIGW-API-KEY", client_secret)

    response = urllib.request.urlopen(request)
    rescode = response.getcode()

    if(rescode==200):
        response_body = response.read()
        result =  json.loads(response_body.decode('utf-8'))
        return result["addresses"][0]["x"], result["addresses"][0]["y"]

    else:
        print("Error Code:" + rescode)
        return None

def reverse_geocode(coords: str, orders="admcode", output="json"):
    url = f"https://naveropenapi.apigw.ntruss.com/map-reversegeocode/v2/gc?coords={coords}&orders={orders}&output={output}"
    request = urllib.request.Request(url)
    request.add_header("X-NCP-APIGW-API-KEY-ID", client_id)
    request.add_header("X-NCP-APIGW-API-KEY", client_secret)

    response = urllib.request.urlopen(request)
    rescode = response.getcode()

    if(rescode==200):
        response_body = response.read()
        result =  json.loads(response_body.decode('utf-8'))
        region = result['results'][0]['region']
        return region['area1']['name'], region['area2']['name'], region['area3']['name']

    else:
        print("Error Code:" + rescode)
        return None

{
'status': {
    'code': 0, 'name': 'ok', 'message': 'done'
    }, 
    'results': [
        {
            'name': 'admcode', 
            'code': {
                'id': '4143056000', 
                'type': 'S', 
                'mappingId': '02430108'
                }, 
            'region': {
                'area0': {
                    'name': 'kr', 
                    'coords': {
                        'center': {
                            'crs': '', 
                            'x': 0.0, 
                            'y': 0.0
                            }}}, 
                'area1': {
                    'name': '경기도', 
                    'coords': {'center': {'crs': 'EPSG:4326', 'x': 127.550802, 'y': 37.4363177}}, 
                    'alias': '경기'
                    }, 
                'area2': {
                    'name': '의왕시',
                    'coords': {'center': {'crs': 'EPSG:4326', 'x': 126.9682786, 'y': 37.3448869}}}, 
                'area3': {
                    'name': '청계동',
                    'coords': {'center': {'crs': 'EPSG:4326', 'x': 126.9960392, 'y': 37.3885389}}}, 
                'area4': {
                    'name': '', 'coords': {'center': {'crs': '', 'x': 0.0, 'y': 0.0}}}}}]
}


def map_js(x, y):
    return """
<script type="text/javascript" src="https://openapi.map.naver.com/openapi/v3/maps.js?ncpClientId=0knhtfya6g"></script>
<div id="map" style="width:100%;height:400px;"></div>

<script>
var mapOptions = {
""" + f"    center: new naver.maps.LatLng({y}, {x})," + """
    zoom: 10
};

var map = new naver.maps.Map('map', mapOptions);
</script>"""

if __name__ == '__main__':
    coords = "126.9960392,37.3885389"
    print(reverse_geocode(coords))

    query = "서울시 성동구 행당동 168-151 4F"
    x, y = (get_geocode(query))

    print(map_js(x=x, y=y))