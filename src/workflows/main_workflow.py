from prefect import flow

from src.workflows.fine_tune_flow import fine_tune_flow
from src.workflows.inference_flow import inference_flow


@flow(name="LLM Pipeline")
def llm_pipeline(
    model_id="Qwen/Qwen2.5-1.5B-Instruct",
    data_path="data/dataset.jsonl",
    output_path="./models/lora",
    run_inference=True,
    port=8000,
):
    """Complete LLM pipeline: fine-tuning and inference"""
    # Run fine-tuning
    fine_tune_result = fine_tune_flow(model_id=model_id, data_path=data_path, output_path=output_path)

    results = {"fine_tuning": fine_tune_result}

    # Run inference if requested
    if run_inference and fine_tune_result["status"] == "success":
        inference_result = inference_flow(model_id=model_id, adapter_path=output_path, port=port)
        results["inference"] = inference_result

    return results


if __name__ == "__main__":
    llm_pipeline()
