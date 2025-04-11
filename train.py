''' -----------------------------------------------------------
File Name: train.py

Created Date: 2024.10.28
Last Edited Date: 2024.10.29

Editor: ge.an (gaeun.an)

Content:
    - LSTM을 활용하여 모델 학습

Changes:
    - 
----------------------------------------------------------- '''

import os
import torch
import torch.nn as nn
import torch.optim as optim
import random
from torch.utils.data import DataLoader
from sklearn.preprocessing import RobustScaler
import numpy as np
from train_common import PriceDataset, LSTM, load_data

def train_model(file_path, model_save_path, seed):
    """
		Content: 주어진 데이터 파일을 활용하여 LSTM 모델을 훈련하고 저장한다.
            
		Args:
            file_path: 데이터 파일 경로
            model_save_path: 학습 모델 저장 경로
      
		Returns:

	"""

    # seed 고정
    #seed = 0
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    
    # 데이터 로드
    target, features, target_scaler = load_data(file_path)
    
    # 모델 학습에 필요한 파라미터 설정
    input_size = features.shape[1] # feature 수 (실제 feautres 수를 기반으로 설정)
    hidden_size = 32 # 은닉층 수
    output_size = 1 # target 수
    num_layers = 2 # LSTM 레이어 수
    batch_size = 8 # 배치
    num_epochs = 200 # 에폭
    learning_rate = 0.001 # 학습률
    
    # 스케일링
    scaler = RobustScaler()
    scaled_features = scaler.fit_transform(features)

    # 훈련 데이터와 테스트 데이터 분리
    train_size = int(len(scaled_features) * 0.8)
    train_data = scaled_features[:train_size]
    train_targets = target[:train_size]
    
    # 시퀀스 길이
    seq_length = min(30, len(train_data) // 10)

    # 데이터셋 생성
    train_dataset = PriceDataset(train_data, train_targets, seq_length)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True) # 데이터로더 설정

    # LSTM 모델 초기화
    model = LSTM(input_size, hidden_size, output_size, num_layers)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model.train() # 모델 학습
    
    for epoch in range(num_epochs):
        epoch_loss = 0 # 에폭 당 손실
        for features, targets in train_loader:
            optimizer.zero_grad() # 기울기 초기화
            outputs = model(features)
            
            # 예측값과 타겟값의 크기
            outputs = outputs.view(-1) # 예측값을 1차원으로 변환
            targets = targets.view(-1) # 타겟값을 1차원으로 변환
            
            loss = criterion(outputs, targets) # 예측값과 실제값 간의 손실 계산
            loss.backward() # 기울기 계산
            optimizer.step() # 모델의 가중치 업데이트
            
            epoch_loss += loss.item() 
            
        if (epoch + 1) % 10 == 0:
            print(f'Seed [{seed}], Epoch [{epoch+1}], Loss: {epoch_loss/len(train_loader):.4f}') # 현재 에폭의 평균 손실

    # 모델 저장
    os.makedirs(model_save_path, exist_ok=True)
    model_file_name = f'model_seed_{seed}.pt'
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'input_size': input_size,
        'hidden_size': hidden_size,
        'output_size': output_size,
        'num_layers': num_layers,
        'scaler': scaler,
        'target_scaler': target_scaler,
        'seq_length': seq_length,
        'seed': seed
    }, os.path.join(model_save_path, model_file_name))

if __name__ == "__main__":
    # 데이터 경로 설정
    data_path = 'C:/pricePredict/data/final_data_demo' 
    target_files = ["깐마늘 대서.csv", "만생양파.csv"]
    seeds = random.sample(range(1, 2000), 1)
    print(f"Selected random seeds: {seeds}")
    
    for file in target_files:
        print(f"\nProcessing file: {file}")
        file_path = os.path.join(data_path, file)
        model_save_path = f'C:/pricePredict/data/final_data_demo/{file[:-4]}'

        for seed in seeds:
            print(f"\nTraining with seed: {seed}")
            train_model(file_path, model_save_path, seed)