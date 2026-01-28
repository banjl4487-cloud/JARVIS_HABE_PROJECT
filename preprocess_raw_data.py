import pandas as pd

# 1. 데이터 로드
df = pd.read_csv('unsmile_train_v1.0.tsv', sep='\t')

# 2. 정상(600개) 및 비정상(400개) 데이터 샘플링
clean_df = df[df['clean'] == 1].sample(n=600, random_state=42)
unclean_df = df[df['clean'] == 0].sample(n=400, random_state=42)

# 3. 데이터 병합 및 섞기
combined_df = pd.concat([clean_df, unclean_df]).sample(frac=1, random_state=42).reset_index(drop=True)

# 4. 통합 데이터셋 저장
combined_df.to_csv('hate_speech_dataset_v1.csv', sep='\t', index=False, encoding='utf-8-sig')

# 5. 결과 확인
print(f"Dataset integration complete. Total rows: {len(combined_df)}")
