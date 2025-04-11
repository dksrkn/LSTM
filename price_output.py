''' -----------------------------------------------------------
File Name: price_output.py

Created Date: 2024.09.27
Last Edited Date: 2024.10.28

Editor: ge.an (gaeun.an)

Content:
    - 품목별 가격과 생산량 

Changes:
    - 날짜별로 총거래물량과 총거래금액만 계산, 거래단위, 도매시장, 도매법인 삭제
----------------------------------------------------------- '''

import os
import pandas as pd

# 데이터 파일 경로
data_path = 'C:/pricePredict/data/price_and_output'

# data_path에 있는 데이터 파일 리스트
file_list = ['garlic.xlsx', 'onion.xlsx']

# 특정 품목에 대한 필터링 조건 설정 (각 품목별 품종 선별)
special_conditions = {
    'garlic.xlsx': ['깐마늘 대서', '깐마늘', '기타마늘'],
    'onion.xlsx': ['만생양파', '기타양파', '양파(일반)', '양파'],
}

# 특정 품목에 대한 필터링 조건 설정 (수입)
import_conditions = {
    'garlic.xlsx': ['마늘(수입)', '깐마늘(수입)'],
    'onion.xlsx': ['양파(수입)']
}

def domestic_products(df, conditions):
    """
        Content: 국내산인 품종을 필터링한다.
        
        Args:
            df(DataFrame): 데이터프레임
            conditions: 필터링 조건 
        
        Returns:
            DataFrame: 조건에 맞는 품종 데이터
        
    """
    if isinstance(conditions, list):
        return df[df['품종'].isin(conditions)]
    return df[df['품종'] == conditions]

def imported_products(df, conditions):
    """
        Content: 수입산인 품종을 필터링한다.
        
        Args:
            df(DataFrame): 데이터프레임
            conditions: 필터링 조건
        
        Returns:
            DataFrame: 조건에 맞는 수입 품종 데이터
        
    """
    if conditions:
        return df[df['품종'].isin(conditions)]
    return pd.DataFrame(columns=df.columns)

def calculate(df):
    """
        Content: 날짜별 총거래물량과 총거래금액을 계산한다.
        
        Args:
            df(DataFrame): 품목 데이터프레임
        
        Returns:
            DataFrame: 날짜, 품목, 품종별 총거래물량, 총거래금액, 단위당 가격(원/kg)
        
    """
    if df.empty:
        return pd.DataFrame()
    
    # 날짜별 총거래물량과 총거래금액 계산
    grouped_df = df.groupby(['DATE', '품목', '품종']).agg({
        '총거래물량': 'sum',
        '총거래금액': 'sum'
    }).reset_index()
    
    # 총거래금액(원/kg) 계산
    grouped_df['총거래금액(원/kg)'] = (grouped_df['총거래금액'] / grouped_df['총거래물량']).round(0)
    return grouped_df[['DATE', '품목', '품종', '총거래물량', '총거래금액', '총거래금액(원/kg)']]

for file in file_list:
    print(f"Processing {file}")
    file_path = os.path.join(data_path +'/data_by_item', file)

    df = pd.read_excel(file_path)
    df = df[['DATE', '도매시장', '총거래물량', '총거래금액', '품목', '품종']] # 필요한 컬럼 선택
    df = df[df['도매시장'] == '서울가락도매'] # 서울가락도매시장 기준으로 데이터 수집
    
    # 국내산 품종 데이터 처리
    domestic_df = pd.DataFrame()
    if file in special_conditions:
        domestic_df = domestic_products(df, special_conditions[file])

    # 수입산 품종 데이터 처리
    import_df = pd.DataFrame()
    if file in import_conditions:
        import_df = imported_products(df, import_conditions[file])

    # 계산
    domestic_stats = calculate(domestic_df)
    import_stats = calculate(import_df)

    # 국내산 데이터와 수입산 데이터 통합
    combined_stats = pd.concat([domestic_stats, import_stats], ignore_index=True)

    # 결과 저장
    if not combined_stats.empty:
        combined_stats = combined_stats.sort_values('DATE')

        item_name = os.path.splitext(file)[0]
        output_path = os.path.join(data_path + '/price_and_output', f'{item_name}_가격&생산량.csv')
        combined_stats.to_csv(output_path, index=False, encoding='utf-8-sig')
        print("완료")
