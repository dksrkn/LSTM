''' -----------------------------------------------------------
File Name: final_data.py

Created Date: 2024.10.04
Last Edited Date: 2024.10.28

Editor: ge.an (gaeun.an)

Content:
    - 최종 데이터
    - 총지수, 품목별 소비자물가지수 추가

Changes:
    - 품목별 생산자물가지수, 총지수 추가
    - 수입물량, 수입거래금액 추가
----------------------------------------------------------- '''

import pandas as pd
import os

# CPI 데이터 로드
cpi_path = 'C:/pricePredict/data/price_index/cpi.csv'
cpi_df = pd.read_csv(cpi_path, encoding='euc-kr')

# CPI 매핑
cpi_mapping = {
    '깐마늘 대서': '마늘',
    '만생양파': '양파',
}

ppi_path = 'C:/pricePredict/data/price_index/ppi.csv'
ppi_df = pd.read_csv(ppi_path, encoding='euc-kr')

ppi_mapping = {
    '깐마늘 대서': '마늘',
    '만생양파': '양파',
}

merged_path = 'C:/pricePredict/data/merged/price_index_exclusion_data'
output_path = 'C:/pricePredict/data/final_data/final_data'

os.makedirs(output_path, exist_ok=True)

for file in os.listdir(merged_path):
    if file.endswith('.csv'):
        crop_df = pd.read_csv(os.path.join(merged_path, file), encoding='utf-8-sig')

        # 각 품목에 대한 소비자물가지수와 총지수 값 추가
        for crop, cpi_item in cpi_mapping.items():
            if crop in file:
                for index, row in crop_df.iterrows(): # 데이터프레임의 각 행 순회
                    date = pd.to_datetime(row['DATE']) # 'DATE' 컬럼을 datetime으로 변환
                    date_col = f"{date.year}.{date.month:02d} 월" # 소비자물가지수 데이터의 날짜 형식에 맞게 변환

                    # 해당 월에 대한 소비자물가지수 값이 있으면 추가
                    if date_col in cpi_df.columns:
                        cpi_value = cpi_df.loc[cpi_df['품목별'] == cpi_item, date_col].values # 해당 품목의 소비자물가지수 값
                        total_value = cpi_df.loc[cpi_df['품목별'] == '총지수', date_col].values # 총지수 값
                        if cpi_value.size > 0:
                            crop_df.loc[index, 'CPI'] = cpi_value[0]
                        if total_value.size > 0:
                            crop_df.loc[index, '총지수'] = total_value[0]

        # 각 품목에 대한 생산자물가지수와 총지수 값 추가
        for crop, ppi_item in ppi_mapping.items():
            if crop in file:
                for index, row in crop_df.iterrows(): # 데이터프레임의 각 행 순회
                    date = pd.to_datetime(row['DATE']) # 'DATE' 컬럼을 datetime으로 변환
                    date_col = f"{date.year}.{date.month:02d} 월" # 소비자물가지수 데이터의 날짜 형식에 맞게 변환

                    # 해당 월에 대한 생산자물가지수 값이 있으면 추가
                    if date_col in ppi_df.columns:
                        ppi_value = ppi_df.loc[ppi_df['계정코드별'] == ppi_item, date_col].values # 해당 계정코드의 생산자물가지수 값
                        total_value = ppi_df.loc[ppi_df['계정코드별'] == '총지수', date_col].values # 총지수 값
                        if ppi_value.size > 0:
                            crop_df.loc[index, 'PPI'] = ppi_value[0]
                        if total_value.size > 0:
                            crop_df.loc[index, '총지수_PPI'] = total_value[0]  

        # 'DATE' 컬럼을 datetime 형식으로 변환하고 필터링
        crop_df['DATE'] = pd.to_datetime(crop_df['DATE'])
        crop_df = crop_df[(crop_df['DATE'] >= '2020-01-01') & (crop_df['DATE'] <= '2024-10-31')]

        # 필요한 컬럼 순서로 재설정
        crop_df = crop_df.reindex(['no', 'DATE', 'temp', 'hum', 'rain', 'wind', 'sun_Qy', 'CPI', '총지수', 'PPI', '총지수_PPI', '총거래물량', '총거래금액(원/kg)', '수입물량', '수입거래금액(원/kg)'], axis=1)

        crop_df.to_csv(os.path.join(output_path, f'{file}'), index=False, encoding='utf-8-sig')

print("saved")
