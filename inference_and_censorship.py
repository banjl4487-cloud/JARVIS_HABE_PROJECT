import sqlite3
import pandas as pd
import torch
import re
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from tqdm.auto import tqdm

# 1. 경로 설정 (로컬/서버 환경에 맞춰 상대 경로로 수정)
BASE_MODEL_PATH = "./models/base_model"  # 베이스 모델 폴더
ADAPTER_PATH = "./models/final_danger_model"  # 학습된 LoRA 어댑터 폴더
DATASET_PATH = "./data/llm_dataset.tsv"  # 분석 대상 데이터셋
DB_NAME = "./results/censorship_final_results.db"  # 결과 저장 DB


# 2. 모델 및 토크나이저 로드
def load_model_and_tokenizer():
    print("🔄 모델 및 토크나이저 로딩 중...")

    # 한국어 최적화 토크나이저 로드
    tokenizer = AutoTokenizer.from_pretrained(
        "Bllossom/llama-3.2-Korean-Bllossom-3B",
        trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token

    # 베이스 모델 로드 (메모리 효율을 위해 bfloat16 권장)
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        local_files_only=True
    )

    # 학습된 어댑터(LoRA) 연결
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    model.eval()

    return model, tokenizer


# 3. 데이터베이스 초기화 및 테이블 생성
def init_database():
    os.makedirs(os.path.dirname(DB_NAME), exist_ok=True)
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()
    cur.execute('''
        CREATE TABLE IF NOT EXISTS censorship_report (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            comment TEXT,
            gender_bias REAL,
            social_bias REAL,
            hate_speech REAL,
            harmful_info REAL,
            status TEXT
        )
    ''')
    conn.commit()
    return conn, cur


# 4. 분석 실행 메인 함수
def run_censorship_pipeline():
    # 데이터 로드 확인
    if not os.path.exists(DATASET_PATH):
        print(f"🚨 [에러] 데이터셋 파일을 찾을 수 없습니다: {DATASET_PATH}")
        return

    df = pd.read_csv(DATASET_PATH, sep='\t')

    # 분석 시작 인덱스 설정 (필요에 따라 수정)
    start_index = 0
    df_todo = df.iloc[start_index:].copy()

    model, tokenizer = load_model_and_tokenizer()
    conn, cur = init_database()

    # 배치 처리 설정 (GPU 효율 극대화)
    batch_size = 16
    print(f"🔥 분석 시작: 총 {len(df_todo)}건 데이터 처리 중...")

    for i in tqdm(range(0, len(df_todo), batch_size)):
        batch_df = df_todo.iloc[i: i + batch_size]
        batch_texts = batch_df['comments'].tolist()

        # 프롬프트 구성: 항목별 위험도 수치 도출
        prompts = [f"문장: {txt}\n항목별 위험도 점수(0-100)만 숫자로 알려줘: 성별편향, 사회차별, 언어폭력, 범죄유해" for txt in batch_texts]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)

        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=50)
            responses = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        for idx, response in enumerate(responses):
            # 정규표현식을 사용하여 숫자 데이터만 추출
            res_split = response.split("범죄유해")[-1]
            scores = re.findall(r'\d+', res_split)

            # 추출된 데이터 정제 (실패 시 0.0 처리)
            g, s, h, i_score = map(float, scores[:4]) if len(scores) >= 4 else (0.0, 0.0, 0.0, 0.0)

            # 임계값(Threshold) 30점 기준 위험도 판별
            status = "DANGER" if max(g, s, h, i_score) >= 30 else "SAFE"

            # DB 저장
            cur.execute('''
                INSERT INTO censorship_report (comment, gender_bias, social_bias, hate_speech, harmful_info, status)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (batch_texts[idx], g, s, h, i_score, status))

        # 데이터 유실 방지를 위한 주기적 Commit
        if (i // batch_size + 1) % 10 == 0:
            conn.commit()

    conn.commit()
    conn.close()
    print(f"🎉 작업 완료! 결과가 {DB_NAME}에 성공적으로 저장되었습니다.")


if __name__ == "__main__":
    run_censorship_pipeline()