''' -----------------------------------------------------------
File Name: mapping.py

Created Date: 2024.09.24
Last Edited Date: 2024.09.25

Editor: ge.an (gaeun.an)

Content:
    - 품몰 별 주산지 선정 

Changes:
    - ge.an (gaeun.an): 품목 별 kg당 거래물량 계산하여 주산지 선정하도록 변경
    - ge.an (2024.10.04): 품목 별 kg당 거래물량이 아닌, 총거래물량만을 가지고 주산지 선정하도록 변경
----------------------------------------------------------- '''

import os
import pandas as pd
import re

#데이터 파일 경로
data_path = 'C:/pricePredict/data/mapping'

#data_path에 있는 데이터 파일 리스트
file_list = ['garlic.xlsx', 'onion.xlsx']

#특정 품목에 대한 필터링 조건 설정(각 품목 별 품종 선별)
special_conditions = {
    'garlic.xlsx': ['깐마늘 대서', '햇마늘 대서', '깐마늘 남도'],
    'onion.xlsx': ['조생양파', '중생양파', '만생양파'],
} 

#최종 데이터 넣을 빈 프레임으로 초기화
final_df = pd.DataFrame()

for file in file_list:
    file_path = os.path.join(data_path, file)
    print(f"{file}")
    
    df = pd.read_excel(file_path)
    df = df[['거래단위', '총거래물량', '도매시장', '도매법인', '품목', '품종', '산지-광역시도', '산지-시군구']] #필요한 컬럼 선택
    df = df[df['도매시장'] == '서울가락도매'] #서울도매가락시장 기준으로 데이터 수집
    
    #특정 품종에 대해 필터링 조건 적용 (special_conditions)
    if file in special_conditions:
        condition = special_conditions[file]
        if isinstance(condition, list): #조건이 리스트일 경우
            df = df[df['품종'].isin(condition)]
        else: #조건이 단일일 경우
            df = df[df['품종'] == condition]
            
    df = df.sort_values(by='총거래물량', ascending=False).drop_duplicates(subset=['산지-광역시도', '산지-시군구']) #산지 중복 제거 후 총거래물량이 가장 큰 값 선택
    
    final_df = pd.concat([final_df, df]) #최종 데이터 프레임에 추가

#csv 파일로 저장
output_csv_path = os.path.join(data_path, '서울가락도매시장.csv')
final_df.to_csv(output_csv_path, index=False, encoding='utf-8-sig')

print(f"'{output_csv_path}'에 저장")

#주산지 선정을 위한 데이터 프레임 복사
final_production_df = final_df.copy()

#주산지 최종 데이터 넣을 빈 프레임으로 초기화
primary_production_df = pd.DataFrame()

#각 품목 별 주산지 선정 (품종 별로 총거래물량이 가장 큰 값을 주산지로 선정)
for item in final_production_df['품목'].unique():
    if item in ['마늘', '양파', '얼갈이배추', '포도', '상추']:
        filtered_df = final_production_df[final_production_df['품목'] == item]
        if item == '마늘':
            for variety in ['깐마늘 대서', '햇마늘 대서', '깐마늘 남도']:
                variety_df = filtered_df[filtered_df['품종'] == variety]
                if not variety_df.empty:
                    primary_production = variety_df.loc[variety_df['총거래물량'].idxmax()]
                    primary_production_df = pd.concat([primary_production_df, primary_production.to_frame().T], ignore_index=True)

        elif item == '양파':
            for variety in ['조생양파', '중생양파', '만생양파']:
                variety_df = filtered_df[filtered_df['품종'] == variety]
                if not variety_df.empty:
                    primary_production = variety_df.loc[variety_df['총거래물량'].idxmax()]
                    primary_production_df = pd.concat([primary_production_df, primary_production.to_frame().T], ignore_index=True)

        elif item == '얼갈이배추':
            for variety in ['얼갈이배추', '알배기']:
                variety_df = filtered_df[filtered_df['품종'] == variety]
                if not variety_df.empty:
                    primary_production = variety_df.loc[variety_df['총거래물량'].idxmax()]
                    primary_production_df = pd.concat([primary_production_df, primary_production.to_frame().T], ignore_index=True)
        
        elif item == '포도':
            for variety in ['캠벨얼리', '샤인마스캇']:
                variety_df = filtered_df[filtered_df['품종'] == variety]
                if not variety_df.empty:
                    primary_production = variety_df.loc[variety_df['총거래물량'].idxmax()]
                    primary_production_df = pd.concat([primary_production_df, primary_production.to_frame().T], ignore_index=True)

        elif item == '상추':
            for variety in ['청상추', '적상추']:
                variety_df = filtered_df[filtered_df['품종'] == variety]
                if not variety_df.empty:
                    primary_production = variety_df.loc[variety_df['총거래물량'].idxmax()]
                    primary_production_df = pd.concat([primary_production_df, primary_production.to_frame().T], ignore_index=True)
    else:
        item_df = final_production_df[final_production_df['품목'] == item]
        if not item_df.empty:
            primary_production = item_df.loc[item_df['총거래물량'].idxmax()]
            primary_production_df = pd.concat([primary_production_df, primary_production.to_frame().T], ignore_index=True)

#csv 파일로 저장
output_primary_csv_path = os.path.join(data_path, '주산지.csv')
primary_production_df.to_csv(output_primary_csv_path, index=False, encoding='utf-8-sig')

print(f"'{output_primary_csv_path}'에 저장")
