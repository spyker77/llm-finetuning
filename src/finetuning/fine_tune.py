import os

import mlflow
from datasets import load_dataset
from loguru import logger
from peft import LoraConfig, get_peft_model  # type: ignore
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer


def convert_to_chatml(example):
    instr = example.get("instruction", "")
    inp = example.get("input", "")
    out = example.get("output", "")
    if inp:
        text = f"User: {instr} {inp}\nAssistant: {out}"
    else:
        text = f"User: {instr}\nAssistant: {out}"
    return {"text": text}


# Configure MLflow
mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"))
mlflow.set_experiment("qwen2.5-finetuning")


# Start MLflow run
with mlflow.start_run(run_name="qwen2.5-lora"):
    # Log parameters
    mlflow.log_params(
        {
            "model": "Qwen/Qwen2.5-1.5B-Instruct",
            "technique": "LoRA with MLX",
            "r": 8,
            "lora_alpha": 32,
            "learning_rate": 1e-5,
        }
    )

    # Load dataset
    dataset = load_dataset("json", data_files="data/dataset.jsonl")
    dataset = dataset.map(convert_to_chatml)

    # Load model and tokenizer
    model_id = "Qwen/Qwen2.5-1.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Use device='mps' for Apple Silicon GPU acceleration
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")

    # Configure LoRA
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # Apply LoRA
    model = get_peft_model(model, lora_config)

    # Set up training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        warmup_steps=10,
        max_steps=500,
        learning_rate=1e-5,
        logging_steps=10,
        save_steps=100,
        report_to="mlflow",
    )

    # Set up trainer
    trainer = SFTTrainer(model=model, args=training_args, train_dataset=dataset["train"], peft_config=lora_config)

    # Train model
    trainer.train()

    # Save model
    model_path = "./models/lora"
    trainer.model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)

    # Log model
    mlflow.log_artifacts(model_path, "model")

    logger.info("Training complete!")
