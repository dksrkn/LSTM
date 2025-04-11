''' -----------------------------------------------------------
File Name: weather.py

Created Date: 2024.09.30
Last Edited Date: 2024.09.30

Editor: ge.an (gaeun.an)

Content:
    - 공공데이터포털 API를 활용하여 각 주산지별 기상 데이터 수집

Changes:
    - 
----------------------------------------------------------- '''

import pandas as pd
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from tqdm import tqdm

def get_weather_data(service_key, stn_code, stn_name, year, month):
    """
		Content: 
            API에서 특정 관측지점, 연도 및 월의 기상 데이터를 가져온다.
            
		Args:
		    service_key: 공공데이터포털 API 키
            stn_code: 관측지점코드
            stn_name: 관측지점명
            year: 연도
            month: 월
		
		Returns:
		    items: 요청이 성공한 경우 XML 불러오기
            None: 요청이 실패한 경우에는 예외, 그렇지 않은 경우에는 None
    """
    url = 'http://apis.data.go.kr/1390802/AgriWeather/WeatherObsrInfo/V2/GnrlWeather/getWeatherMonDayList?'
    params = {
        'serviceKey': service_key,  
        'Page_No': '1',  
        'Page_Size': '31',
        'search_Year': year,  
        'search_Month': month,  
        'obsr_Spot_Nm': urllib.parse.quote(stn_name),
        'obsr_Spot_Code': stn_code  
    }
    
    for key, value in zip(params.keys(), params.values()):
        if key == 'serviceKey' :
            url = url + key +'=' + value
        else :
            url = url + '&' + key + '=' + value
                    
    try:
        response = urllib.request.urlopen(url).read()
        response_string = ET.fromstring(response)
        items = response_string[1][3] 
        return items, None
    
    except Exception as e:
        return None,e

def collect_weather_data(service_key, obsr_Spot_Code_list, obsr_Spot_Nm_list, year_list, month_list):
    """
		Content: 
            관측지점, 연, 월에 대한 기상 데이터를 수집하고 오류를 처리한다.
		
		Args:
		    service_key: 공공데이터포털 API 키
            obsr_Spot_Code_list: 관측지점코드 리스트
            obsr_Spot_Nm_list: 관측지점명 리스트
            year_list: 연도 리스트
            month_list: 월 리스트
		
		Returns:
            colname_dict: 수집된 기상 데이터
            errors: 수집 과정에서 발생한 오류 로그
		    
	"""
    colname_dict = dict()
    first_run = False
    errors = {
        'year': [], 'month': [], 'stn_code': [], 'stn_name': [], 'url': [], 'f_obs_date': []
    }
    
    for stn_code, stn_name in tqdm(zip(obsr_Spot_Code_list, obsr_Spot_Nm_list)):
        for year in year_list:
            for month in month_list:
                items, error = get_weather_data(service_key, stn_code, stn_name, year, month)

                if error:
                    errors['year'].append(year)
                    errors['month'].append(month)
                    errors['stn_code'].append(stn_code)
                    errors['stn_name'].append(stn_name)
                    errors['url'].append(f"Failed URL for {stn_name}")
                    errors['f_obs_date'].append("")
                    continue  

                if not first_run and items is not None:
                    for i in items[0]:
                        colname_dict[i.tag] = []
                    first_run = True

                if items is not None:
                    num_days = len(items)
                    for index in range(num_days):
                        for i in items[index]:
                            colname_dict[i.tag].append(i.text)

    return colname_dict, errors

def save_weather_data(colname_dict, output_path):
    """
		Content: 
            수집된 기상 데이터를 csv 파일로 저장한다.
            
		Args:
		    colname_dict: 수집된 기상 데이터
            output_path: csv 파일 저장 경로
		
		Returns:
            weather: 기상 데이터를 담은 데이터 프레임
		    
	"""
    weather = pd.DataFrame(colname_dict)
    weather.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"weather_data 저장 완료")
    return weather

def save_error_log(errors, error_path):
    """
		Content: 
            데이터 수집 과정 중 발생한 오류 로그를 csv 파일로 저장한다.
            
		Args:
		    errors: 발생한 오류 로그
            error_path: csv 파일 저장 경로
		
		Returns:
		    None
	"""
    error_df = pd.DataFrame(errors)
    error_df.to_csv(error_path, index=False, encoding='utf-8-sig')
    print(f"error_log 저장 완료")
    
def main():
    """
		Content: 
            기상 데이터 및 오류 로그 수집 및 저장 프로세스 실행한다.
            
		Args:
		    None
      
		Returns:
		    None
	"""
    service_key = open('./data/weather/api.txt', 'r').read().strip() #일반 인증키(Encoding)
    observatory_df = pd.read_csv('./data/mapping/observatory.csv')
    obsr_Spot_Nm_list = observatory_df['관측지점명'].tolist()
    obsr_Spot_Code_list = observatory_df['관측지점코드'].tolist()

    year_list = ['2020', '2021', '2022', '2023', '2024']
    month_list = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']

    colname_dict, errors = collect_weather_data(service_key, obsr_Spot_Code_list, obsr_Spot_Nm_list, year_list, month_list)

    save_weather_data(colname_dict, 'C:/pricePredict/data/weather/weather.csv')
    save_error_log(errors, 'C:/pricePredict/data/weather/error_log.csv')

if __name__ == '__main__':
    main()
