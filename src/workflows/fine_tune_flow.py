import os

import mlflow
from datasets import load_dataset
from peft import LoraConfig, get_peft_model  # type: ignore
from prefect import flow, task
from prefect.cache_policies import NO_CACHE
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer


@task(cache_policy=NO_CACHE)
def prepare_dataset(data_path):
    """Load and prepare the dataset for fine-tuning"""
    dataset = load_dataset("json", data_files=data_path)

    def convert_to_chatml(example):
        instr = example.get("instruction", "")
        inp = example.get("input", "")
        out = example.get("output", "")
        if inp:
            text = f"User: {instr} {inp}\nAssistant: {out}"
        else:
            text = f"User: {instr}\nAssistant: {out}"
        return {"text": text}

    return dataset.map(convert_to_chatml)


@task(cache_policy=NO_CACHE)
def initialize_model(model_id):
    """Initialize the model and tokenizer"""
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")
    return model, tokenizer


@task(cache_policy=NO_CACHE)
def configure_lora(model, r, alpha):
    """Configure and apply LoRA to the model"""
    lora_config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
    )

    return get_peft_model(model, lora_config), lora_config


@task(cache_policy=NO_CACHE)
def train_model(model, tokenizer, dataset, lora_config, learning_rate, max_steps):
    """Train the model using SFTTrainer"""
    training_args = TrainingArguments(
        output_dir="./results",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        warmup_steps=10,
        max_steps=max_steps,
        learning_rate=learning_rate,
        logging_steps=10,
        save_steps=100,
        report_to="mlflow",
    )

    trainer = SFTTrainer(model=model, args=training_args, train_dataset=dataset["train"], peft_config=lora_config)

    trainer.train()
    return trainer


@task(cache_policy=NO_CACHE)
def save_model(trainer, tokenizer, output_path):
    """Save the fine-tuned model"""
    trainer.model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    return output_path


@task(cache_policy=NO_CACHE)
def log_to_mlflow(model_path, experiment_name, run_name, params):
    """Log the model and parameters to MLflow"""
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(params)
        mlflow.log_artifacts(model_path, "model")

    return True


@flow(name="LLM Fine-tuning Flow")
def fine_tune_flow(
    model_id="Qwen/Qwen2.5-1.5B-Instruct",
    data_path="data/dataset.jsonl",
    output_path="./models/lora",
    experiment_name="qwen2.5-finetuning",
    run_name="qwen2.5-lora",
    lora_r=8,
    lora_alpha=32,
    learning_rate=1e-5,
    max_steps=500,
):
    """Orchestrate the fine-tuning process with Prefect"""
    # Prepare parameters for logging
    params = {
        "model": model_id,
        "technique": "LoRA",
        "r": lora_r,
        "lora_alpha": lora_alpha,
        "learning_rate": learning_rate,
        "max_steps": max_steps,
    }

    # Execute the workflow
    dataset = prepare_dataset(data_path)
    model, tokenizer = initialize_model(model_id)
    model_with_lora, lora_config = configure_lora(model, r=lora_r, alpha=lora_alpha)
    trainer = train_model(
        model_with_lora,
        tokenizer,
        dataset,
        lora_config,
        learning_rate=learning_rate,
        max_steps=max_steps,
    )
    saved_path = save_model(trainer, tokenizer, output_path)
    log_to_mlflow(saved_path, experiment_name, run_name, params)

    return {"status": "success", "model_path": saved_path}


if __name__ == "__main__":
    fine_tune_flow()
