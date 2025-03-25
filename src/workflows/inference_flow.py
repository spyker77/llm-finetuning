import subprocess
import time
from datetime import timedelta

import requests
from loguru import logger
from prefect import flow, task
from prefect.tasks import task_input_hash


@task(cache_key_fn=task_input_hash, cache_expiration=timedelta(days=1))
def start_inference_server(model_id, adapter_path, port):
    """Start the FastAPI inference server"""
    # Check if the server is already running
    try:
        result = subprocess.run(["lsof", "-i", f":{port}"], capture_output=True, text=True)
        if result.stdout:
            logger.info(f"Server already running on port {port}")
            return {"status": "already_running", "port": port}
    except Exception:
        pass

    # Start the server as a background process
    cmd = [
        "python",
        "src/inference/inference.py",
        "--model-id",
        model_id,
        "--adapter-path",
        adapter_path,
        "--port",
        str(port),
    ]

    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    # Give the server time to start
    time.sleep(5)

    # Check if the server started successfully
    if process.poll() is None:
        return {"status": "started", "port": port, "pid": process.pid}
    else:
        stdout, stderr = process.communicate()
        raise RuntimeError(f"Failed to start inference server: {stderr}")


@task
def test_inference_endpoint(port):
    """Test the inference endpoint with a simple query"""
    url = f"http://localhost:{port}/generate"
    payload = {"prompt": "Hello, how are you?", "max_length": 50, "temperature": 0.7, "top_p": 0.9}

    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        return {"status": "success", "response": response.json(), "endpoint": url}
    except Exception as e:
        return {"status": "error", "error": str(e), "endpoint": url}


@flow(name="LLM Inference Flow")
def inference_flow(model_id="Qwen/Qwen2.5-1.5B-Instruct", adapter_path="models/lora", port=8000):
    """Start and test the inference server"""
    server_status = start_inference_server(model_id, adapter_path, port)
    if server_status["status"] in ["started", "already_running"]:
        test_result = test_inference_endpoint(port)
        return {"server_status": server_status, "test_result": test_result}
    else:
        return {"status": "error", "details": server_status}


if __name__ == "__main__":
    inference_flow()
