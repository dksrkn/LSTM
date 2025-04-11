''' -----------------------------------------------------------
File Name: train_test.py

Created Date: 2024.10.28
Last Edited Date: 2024.10.29

Editor: ge.an (gaeun.an)

Content:
    - train.py에서 학습한 모델을 활용하여 가격 예측

Changes:
    - 각 seed별 모델의 예측 결과를 개별적으로 저장하는 기능 추가
----------------------------------------------------------- '''

import os
import torch
import pandas as pd
from train_common import LSTM, load_data, save_predictions
import numpy as np

def load_model(model_path):
    """
        Content: 학습시킨 모델을 불러온다.
            
        Args:
            model_path: 학습시킨 모델 경로
      
        Returns:
            tuple: 모델과 체크포인트 정보

    """
    checkpoint = torch.load(model_path, weights_only=False)
    model = LSTM(
        checkpoint['input_size'],
        checkpoint['hidden_size'],
        checkpoint['output_size'],
        checkpoint['num_layers']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    return model, checkpoint

def predict_specific_week(model, last_sequence, scaler, target_scaler, start_date, end_date):
    """
        Content: 특정 주 가격을 예측한다.
            
        Args:
            model: 학습된 LSTM 모델
            last_sequence: 마지막 입력 시퀀스
            scaler: 입력 데이터의 스케일러
            target_scaler: 타켓 데이터의 스케일러
            start_date: 예측 시작일(YYYY-MM-DD)
            end_date: 예측 종료일(YYYY-MM-DD)
      
        Returns:
            tuple: 예측된 가격 리스트와 날짜 범위

    """
    model.eval()
    predictions = []
    current_sequence = last_sequence.copy()
    
    date_range = pd.date_range(start=start_date, end=end_date)

    for date in date_range:
        with torch.no_grad():
            input_tensor = torch.tensor(current_sequence, dtype=torch.float32).unsqueeze(0)
            predicted = model(input_tensor)
            predicted = predicted.numpy()

            predicted_price = target_scaler.inverse_transform(predicted.reshape(-1, 1))[0][0]
            predictions.append(predicted_price)

            new_features = current_sequence[-1].copy()
            new_features[0] = date.year
            new_features[1] = date.month
            new_features[2] = date.day

            current_sequence = np.vstack([current_sequence[1:], new_features])
    
    return predictions, date_range

def save_individual_predictions(file_name, predictions, date_range, save_path, seed):
    """
    Content: 개별 모델의 예측 결과를 저장한다.
    
    Args:
        file_name: 데이터 파일 이름
        predictions: 예측 결과
        date_range: 예측 날짜 범위
        save_path: 저장 경로
        seed: 모델의 seed 값
        
    Returns:
        DataFrame: 예측 결과 데이터프레임
    """
    os.makedirs(save_path, exist_ok=True)
    
    pred_df = pd.DataFrame({
        'date': date_range,
        'predicted_price': predictions
    })
    
    save_file = os.path.join(save_path, f'{file_name}_seed_{seed}.csv')
    pred_df.to_csv(save_file, index=False)
    print(f"Predictions saved for seed {seed} at: {save_file}")
    
    return pred_df

def predict_prices(file_path, model_dir, start_date, end_date, save_path='C:/pricePredict/data/final_data_demo'):
    """
        Content: 가격을 예측하고 예측 결과를 저장한다.
            
        Args:
            file_path: 데이터 파일 경로
            model_dir: 모델 디렉토리 경로
            start_date: 예측 시작일(YYYY-MM-DD)
            end_date: 예측 종료일(YYYY-MM-DD)
            save_path: 예측 결과 저장 경로
            
        Returns:
            list: 각 모델별 예측 결과 데이터프레임 리스트

    """
    _, features, _ = load_data(file_path)

    model_files = [f for f in os.listdir(model_dir) if f.startswith('model_seed_') and f.endswith('.pt')]
    
    prediction_dfs = []
    all_predictions = []
    file_name = os.path.basename(file_path)[:-4]
    
    for model_file in model_files:
        seed = model_file.split('_')[-1].replace('.pt', '')
        
        model_path = os.path.join(model_dir, model_file)
        model, checkpoint = load_model(model_path)
        scaler = checkpoint['scaler']
        target_scaler = checkpoint['target_scaler']
        seq_length = checkpoint['seq_length']

        scaled_features = scaler.transform(features)
        last_sequence = scaled_features[-seq_length:]

        predictions, date_range = predict_specific_week(
            model, last_sequence, scaler, target_scaler, start_date, end_date
        )
        
        pred_df = save_individual_predictions(file_name, predictions, date_range, save_path, seed)
        prediction_dfs.append(pred_df)
        all_predictions.append(predictions)
    
    return prediction_dfs

if __name__ == "__main__":
    data_path = 'C:/pricePredict/data/final_data_demo'
    target_files = ["깐마늘 대서.csv", "만생양파.csv"] 
    start_date = '2024-12-01'
    end_date = '2024-12-23'
    
    for file in target_files:
        print(f"\nProcessing file: {file}")
        file_path = os.path.join(data_path, file)
        model_dir = f'C:/pricePredict/data/final_data_demo/{file[:-4]}'
        predictions = predict_prices(file_path, model_dir, start_date, end_date)
        print(f"Completed predictions for: {file}")
