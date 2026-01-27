import pandas as pd

# 1. 원본 데이터 로드
# TSV 파일이므로 탭(\t) 구분자를 지정하여 읽어옵니다.
train_df = pd.read_csv('train.tsv', sep='\t')
dev_df = pd.read_csv('dev.tsv', sep='\t')

# 2. 데이터 병합
# 두 데이터를 하나로 합치고, ignore_index를 통해 인덱스를 재정렬합니다.
combined_df = pd.concat([train_df, dev_df], ignore_index=True)

# 3. 통합 데이터 저장
# 분석 및 검열 작업에 사용할 마스터 데이터셋을 TSV 형식으로 저장합니다.
combined_df.to_csv('llm_dataset.tsv', sep='\t', index=False, encoding='utf-8-sig')

# 4. 결과 확인
print(f"Dataset integration complete. Total rows: {len(combined_df)}")