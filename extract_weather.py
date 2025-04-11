''' -----------------------------------------------------------
File Name: extract_weather.py

Created Date: 2024.10.02
Last Edited Date: 2024.10.02
Editor: ge.an (gaeun.an)

Content:
    - 관측지점별 기상 데이터 csv 파일에서 필요한 컬럼만 선택

Changes:
    - 
----------------------------------------------------------- '''

import pandas as pd
import os

input_file_path = 'C:\pricePredict\data\weather\weather.csv' # 관측지점별 기상 데이터 csv 파일 경로
output_directory = 'C:\pricePredict\data\weather' # 저장 경로

weather_data = pd.read_csv(input_file_path, encoding='utf-8')

selected_columns = ['stn_Code', 'date', 'temp', 'hum', 'wind', 'rain', 'sun_Qy'] # 필요한 컬럼 선택
weather_data = weather_data[selected_columns]

for stn_code, group in weather_data.groupby('stn_Code'):
    output_file_path = os.path.join(output_directory, f'{stn_code}_weather.csv')
    group.to_csv(output_file_path, index=False, encoding='utf-8-sig')

print("success")
