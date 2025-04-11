''' -----------------------------------------------------------
File Name: train_commom.py

Created Date: 2024.10.28
Last Edited Date: 2024.10.28

Editor: ge.an (gaeun.an)

Content:
    - 가격 예측을 위한 데이터셋 및 LSTM 모델 정의
    - 데이터 로드 함수 (train.py, train_test.py에서 사용)
    - 예측 결과 저장 함수 (train_test.py에서 사용)

Changes:
    - 
----------------------------------------------------------- '''

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from sklearn.preprocessing import RobustScaler
import os
from datetime import datetime

class PriceDataset(Dataset):
    """
        Content: target와 feature 데이터를 이용해 학습을 위한 데이터셋을 생성한다.
        
        Attributes:
            features: 에측을 위한 feature 값
            target: 예측하고자 하는 target 값
            seq_length: LSTM 시퀀스 길이
            
        Methods:
            __len__: 데이터셋의 총 길이 반환
            __getitem__: 해당 인덱스에서 LSTM에 사용할 입력 데이터(x)와 타겟 값(y) 반환
    """
    def __init__(self, features, targets, seq_length):
        self.features = features
        self.targets = targets
        self.seq_length = seq_length
    
    def __len__(self):
        return len(self.features) - self.seq_length
    
    def __getitem__(self, idx):
        x = self.features[idx:idx+self.seq_length]
        y = self.targets[idx+self.seq_length]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

class LSTM(nn.Module):
    """
        Content: LSTM 구조를 정의하고 forward 함수를 통해 예측값을 출력한다.
        
        Attributes:
            lstm: LSTM 레이어
            fc: fully connected 레이어
            
        Methods:
            forward: LSTM의 출력값을 통해 최종 예측값 도출
    """
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x, _ = self.lstm(x)
        out = self.fc(x[:, -1, :]) # 시퀀스의 마지막 값을 예측에 사용
        return out

def load_data(file_path):
    """
		Content: 
            각 품종별 csv 데이터 파일을 로드한다.
            필요한 feature와 target 데이터를 추출하여 반환한다.
            
		Args:
		    file_path: csv 파일 경로
      
		Returns:
		    target: 예측하고자 하는 것 (총거래금액(원/kg))
            features_data: 예측 기준이 되는 것
            target_scaler: target 값에 적용된 스케일러
	"""
    encodings = ['utf-8', 'utf-8-sig', 'euc-kr', 'cp949']
    df = None

    for enc in encodings:
        try:
            df = pd.read_csv(file_path, encoding=enc)
            break  
        except Exception as e:
            print(f"Error loading {file_path} - {enc}: {e}")
            
    # 'DATE' 열을 datetime으로 변환하고, 날짜별로 값 정렬
    df['DATE'] = pd.to_datetime(df['DATE'])
    df = df.sort_values(by='DATE')

    # 'DATE' 열을 'Year', 'Month', 'Day'로 분할
    df['Year'] = df['DATE'].dt.year
    df['Month'] = df['DATE'].dt.month
    df['Day'] = df['DATE'].dt.day

    # target 값 스케일링
    target = df['총거래금액(원/kg)'].values
    target_scaler = RobustScaler()
    target = target_scaler.fit_transform(target.reshape(-1, 1)).flatten()
    
    features = ['Year', 'Month', 'Day', 'temp', 'hum', 'rain', 'CPI', '총지수', 'PPI', '총지수_PPI', '총거래물량', '평균_원/달러']
    if 'sun_Qy' in df.columns:
        features.append('sun_Qy')
    if '수입물량' in df.columns:
        features.append('수입물량')

    features_data = df[features].values
    
    return target, features_data, target_scaler

def save_predictions(file_name, predictions, date_range, save_path='predictions_test/'):
    """
		Content: 예측 결과를 CSV 파일로 저장한다.
            
		Args:
		    file_name: 저장할 파일 이름
            predictions: 예측값 배열
            date_range: 예측 기간
            save_path: 저장 경로
      
		Returns:
            pd.DataFrame: 저장된 예측값 데이터프레임
	"""
    os.makedirs(save_path, exist_ok=True)

    dates = [date.strftime('%Y-%m-%d') for date in date_range]
    pred_df = pd.DataFrame({
        'Date': dates,
        'Predicted_Price': predictions
    })

    pred_df.to_csv(f'{save_path}/{file_name}_predictions.csv', index=False)
    return pred_df