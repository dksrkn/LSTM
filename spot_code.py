''' -----------------------------------------------------------
File Name: spot_code.py

Created Date: 2024.09.30
Last Edited Date: 2024.09.30

Editor: ge.an (gaeun.an)

Content:
    - 카카오 API를 활용하여 주산지와 가장 가까운 기상관측지점 알아내기

Changes:
    - 
----------------------------------------------------------- '''

import requests
import json
import pandas as pd
from geopy.distance import geodesic

# 관측지점 정보를 담은 CSV 파일을 읽어와 필요한 컬럼만 선택
data = pd.read_csv('C:/pricePredict/data/mapping/관측지점정보.csv') 
data = data[['지점명', '지점코드', '위도', '경도']]

kakao_serviceKey = open('C:/pricePredict/data/mapping/kakao_serviceKey.txt', 'r').read() #카카오 API 서비스 키 파일에서 읽어오기
headers = {"Authorization": f"KakaoAK {kakao_serviceKey}"} #API 요청에 필요한 헤더 정보 설정
locations = ["부산광역시"] #주산지 위치 리스트(확인하고자 하는 주산지 작성)

# 카카오 API의 주소 검색 URL
url = 'https://dapi.kakao.com/v2/local/search/address.json?query='

# 두 좌표 간의 거리를 계산하는 함수 (위도, 경도를 입력받아 km 단위 거리 계산)
def calculate_distance(coord1, coord2):
    return geodesic(coord1, coord2).kilometers

#각 주산지 위치에 대해 가까운 관측지점 찾기
for location in locations:
    request_url = url + location
    
    #카카오 API에 GET 요청을 보내서 주산지 위치 정보 응답 받기
    response = requests.get(request_url, headers=headers)
    result = json.loads(response.text)
    
    #응답 정상 & 'documents'에 결과가 있을 경우
    if 'documents' in result and len(result['documents']) > 0:
        match_first = result['documents'][0]['address'] #첫 번째 검색 결과 사용

        #주산지 위도 경도 
        lat = float(match_first['y']) 
        lon = float(match_first['x'])
        
        print(f"주산지: {location}")
        print(f"위도, 경도: {lat}, {lon}")
        
        #각 주산지와 관측지점 간의 거리를 계산하여 'distance' 컬럼에 저장
        data['distance'] = data.apply(
            lambda row: calculate_distance((lat, lon), (row['위도'], row['경도'])), axis=1
        )
        
        #가장 가까운 관측지점 찾기 (거리 값이 최소인 것)
        closest_station = data.loc[data['distance'].idxmin()]
        
        print(f"{location}의 가장 가까운 관측지점: {closest_station['지점명']}, 코드: {closest_station['지점코드']}, 거리: {closest_station['distance']} km")
