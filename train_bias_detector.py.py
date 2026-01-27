import torch
import os
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model, TaskType

# 1. 환경 설정 및 모델 로드
# Base Model: Bllossom (Llama-3.2-3B 기반 한국어 최적화 모델)
model_id = "Bllossom/llama-3.2-Korean-Bllossom-3B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

# 2. 데이터셋 전처리
# 데이터 구성: 약 8,367건의 한국어 편향성/혐오표현 데이터셋 (TSV 형식)
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=512
    )

dataset = load_dataset("csv", data_files="data/llm_dataset.tsv", delimiter="\t")
tokenized_datasets = dataset.map(tokenize_function, batched=True)

# 3. PEFT(LoRA) 구성
# 효율적인 파인튜닝을 위해 LoRA(Low-Rank Adaptation) 기법 적용
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj"]
)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
model = get_peft_model(model, peft_config)

# 4. 학습 파라미터(Hyperparameters) 설정
training_args = TrainingArguments(
    output_dir="./output/bias-detector-v1",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-5,
    num_train_epochs=3,
    logging_steps=10,
    save_strategy="epoch",
    bf16=True,  # bfloat16 지원 환경 기준
    push_to_hub=False,
    report_to="none"
)

# 5. Trainer 초기화 및 학습 수행
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True)
)

if __name__ == "__main__":
    print("Starting training...")
    trainer.train()

    # 6. 학습 완료 모델 저장
    save_path = "./final_model"
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"Model saved to {save_path}")