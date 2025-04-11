''' -----------------------------------------------------------
File Name: merged_.py

Created Date: 2024.09.27
Last Edited Date: 2024.10.28

Editor: ge.an (gaeun.an)

Content:
    - 품종별 가격&생산량, 기상 데이터 합치기

Changes:
    - 
----------------------------------------------------------- '''

import pandas as pd
import os

price_output_dir = 'C:/pricePredict/data/price_and_output/price_and_output' # 가격&생산량 데이터 경로
weather_dir = 'C:/pricePredict/data/weather/' # 기상 데이터 경로
output_dir = 'C:/pricePredict/data/merged/price_index_exclusion_data' # 저장 경로

# 품목별 기상 관측지점 코드
observatory = {
    '마늘' : '635821A001',
    '양파': '534833E001',
}

# 각 파일에 대해 매칭되는 품종 정의 (국내)
special_conditions = {
    'garlic_가격&생산량_8.csv': ['깐마늘 대서', '깐마늘', '기타마늘'],
    'onion_가격&생산량_8.csv': ['만생양파', '기타양파', '양파(일반)', '양파'],
}

# 각 파일에 대해 매칭되는 품종 정의 (수입)
import_conditions = {
    'garlic_가격&생산량_8.csv': ['마늘(수입)', '깐마늘(수입)'],
    'onion_가격&생산량_8.csv': ['양파(수입)']
}


def process_special_conditions(file_name, item_list, price_df):
    """
		Content: 조건(국내)에 맞는 품종 데이터를 필터링하고, 날짜별 총거래물량과 총거래금액을 합산한다.
            
		Args:
            file_name(str): 처리 중인 파일 이름
            item_list(list or str): 품종 이름 또는 이름 리스트
            price_df(DataFrame): 가격 및 생산량 데이터프레임
      
		Returns:
            DataFrame: 조건에 따른 날짜별 데이터

	"""
    df = price_df[['DATE', '품목', '품종', '총거래물량', '총거래금액', '총거래금액(원/kg)']]
    
    if isinstance(item_list, str):
        item_list = [item_list]

    mask = df['품종'].isin(item_list)
    special_df = df[mask].copy()
  
    special_agg = special_df.groupby('DATE').agg({
        '총거래물량': 'sum',
        '총거래금액': 'sum'
    }).reset_index()
    
    special_agg['총거래금액(원/kg)'] = (special_agg['총거래금액'] / special_agg['총거래물량']).round(0)
    
    return special_agg

def process_import_conditions(file_name, price_df):
    """
		Content: 조건(수입)에 맞는 품종 데이터를 필터링하고, 날짜별 총거래물량과 총거래금액을 합산한ㄷ.
            
		Args:
            file_name(str): 처리 중인 파일 이름
            price_df(DataFrame): 가격 및 생산량 데이터프레임
      
		Returns:
            DataFrame: 조건에 따른 날짜별 데이터

	"""
    if file_name not in import_conditions:
        return None
        
    df = price_df[['DATE', '품목', '품종', '총거래물량', '총거래금액', '총거래금액(원/kg)']]

    mask = df['품종'].isin(import_conditions[file_name])
    import_df = df[mask].copy()
    
    if len(import_df) == 0:
        return None

    import_agg = import_df.groupby('DATE').agg({
        '총거래물량': 'sum',
        '총거래금액': 'sum'
    }).reset_index()
    
    import_agg['수입거래금액(원/kg)'] = (import_agg['총거래금액'] / import_agg['총거래물량']).round(0)
    import_agg = import_agg.rename(columns={'총거래물량': '수입물량'})
    import_agg = import_agg[['DATE', '수입물량', '수입거래금액(원/kg)']]
    
    return import_agg

for file_name, item_list in special_conditions.items():
    # 가격&생산량 데이터 가져오기
    price_file_path = os.path.join(price_output_dir, file_name)
    price_df = pd.read_csv(price_file_path, encoding='utf-8-sig')

    # 각 조건에 따른 데이터 처리
    special_agg = process_special_conditions(file_name, item_list, price_df)
    import_agg = process_import_conditions(file_name, price_df)

    # 품종 매핑 설정
    item_mapping = {
        '깐마늘 대서': '마늘', '깐마늘': '마늘', '기타마늘': '마늘',
        '만생양파': '양파', '기타양파': '양파', '양파(일반)': '양파', '양파': '양파',
    }
    
    # item_list의 첫 품종에 따른 품종 매핑
    item = item_mapping.get(item_list[0] if isinstance(item_list, list) else item_list)
    
    if not item:
        continue
    
    # 해당 품목의 기상 데이터 불러오기
    stn_Code = observatory.get(item)
    weather_file_path = os.path.join(weather_dir, f"{stn_Code}_weather.csv")
    weather_df = pd.read_csv(weather_file_path, encoding='utf-8')
    weather_df.rename(columns={
        'date': 'DATE',
        '기온(°C)': 'temp',
        '강수량(mm)': 'rain',
        '습도(%)': 'hum'
    }, inplace=True)

    # 특수 조건 데이터와 기상 데이터 병합
    merged_df = pd.merge(special_agg, weather_df, on='DATE', how='inner')
    
    # 수입 조건 데이터가 있으면 병합, 없으면 0으로 채움
    if import_agg is not None:
        merged_df = pd.merge(merged_df, import_agg, on='DATE', how='left')
    else:
        merged_df['수입물량'] = 0
        merged_df['수입거래금액(원/kg)'] = 0

    # 최종 컬럼
    final_columns = [
        'DATE', 'temp', 'hum', 'rain', 'wind', 'sun_Qy',
        '총거래물량', '총거래금액(원/kg)',
        '수입물량', '수입거래금액(원/kg)'
    ]
    
    # 최종 데이터 저장
    final_df = merged_df[final_columns]
    final_df.insert(0, 'no', range(1, len(final_df) + 1))

    item_output_file_name = f"{file_name.replace('_가격&생산량_8.csv', '')}_{item}.csv"
    item_output_file_path = os.path.join(output_dir, item_output_file_name)
    final_df.to_csv(item_output_file_path, index=False, encoding='utf-8-sig')
    print(f"완료")