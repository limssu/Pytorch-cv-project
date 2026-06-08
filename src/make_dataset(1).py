// 교수님 피드백 반영 새로운 데이터셋

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split

# 경로 정의
TARGET_PROCESSED_DIR = "./processed_data"
os.makedirs(TARGET_PROCESSED_DIR, exist_ok=True)

print("[INFO] Loading master dataset (LSWMD.pkl). This process may take up to 1 minute.")
df = pd.read_pickle("/home/ubuntu/raw_data/LSWMD.pkl")
print(f"[INFO] Master dataset successfully loaded. Total records: {len(df)} samples.")

# 1. 라벨 데이터 무결성 검증 및 예외 처리 함수
def sanitize_label(label_obj):
    if isinstance(label_obj, np.ndarray):
        if label_obj.size == 0:
            return 'Unknown'
        return str(label_obj[0][0]).strip()
    return str(label_obj).strip()

print("[INFO] Executing label sanitization and noise filtering...")
df['failureType_parsed'] = df['failureType'].apply(sanitize_label)

# 2. 비지도 학습용 순수 정상(none) 데이터셋 추출
df_normal = df[df['failureType_parsed'].str.lower() == 'none'].copy()
print(f"[INFO] Target subset filtering completed. Pure normal data size: {len(df_normal)} samples.")

# 3. 교수님 피드백 반영: Train / Val 비율을 늘리고 Test(정상)는 3,000장으로 축소
# 전체 약 14.7만 장 중 Train(약 11.4만 장), Val(약 3만 장), Test(3,000장) 규모 분할
train_df, remainder_df = train_test_split(df_normal, test_size=0.22, random_state=42)
val_df, test_normal_raw = train_test_split(remainder_df, test_size=0.09, random_state=42)

# 정상 테스트셋을 정확히 3,000장만 무작위 샘플링
test_normal_compact = test_normal_raw.sample(n=3000, random_state=42)

# 4. 실제 공정 불량 데이터(Unknown 제외)에서 실제 비율에 맞춰 600장 무작위 샘플링
df_defect = df[df['failureType_parsed'].str.lower() != 'none'].copy()
df_real_defect = df_defect[df_defect['failureType_parsed'].str.lower() != 'unknown'].copy()
test_defect_compact = df_real_defect.sample(n=600, random_state=42)

# 5. 정상 3,000장 + 불량 600장 혼합하여 최종 실전 시험지(3,600장) 구축 및 셔플
final_test_set = pd.concat([test_normal_compact, test_defect_compact]).sample(frac=1, random_state=42)

# 6. 영구 저장소 (EBS) 내 바이너리 데이터 고속 적재
print("[INFO] Serializing and writing processed subsets to local storage...")
train_df.to_pickle(os.path.join(TARGET_PROCESSED_DIR, 'train_data.pkl'))
val_df.to_pickle(os.path.join(TARGET_PROCESSED_DIR, 'val_data.pkl'))
final_test_set.to_pickle(os.path.join(TARGET_PROCESSED_DIR, 'test_data.pkl'))

# 7. 최종 파이프라인 무결성 리포팅
print("\n=======================================================")
print("             DATA PIPELINE EXECUTION SUMMARY           ")
print("=======================================================")
print(f" * Training Subset     : {len(train_df):>8,} samples (정상)")
print(f" * Validation Subset   : {len(val_df):>8,} samples (정상)")
print(f" * Mixed Test Subset   : {len(final_test_set):>8,} samples (정상 3K + 불량 600장)")
print("-------------------------------------------------------")
print(f" * Total Unified Samples : {len(train_df)+len(val_df)+len(final_test_set):>8,} samples")
print(f" * Target Output Path   : {os.path.abspath(TARGET_PROCESSED_DIR)}")
print("=======================================================")
print("[STATUS] Data pipeline sequence terminated successfully.\n")
