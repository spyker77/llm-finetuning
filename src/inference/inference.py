# from fastapi import FastAPI, HTTPException
# from loguru import logger
# from peft import PeftModel # type: ignore
# from pydantic import BaseModel
# from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# app = FastAPI()


# class Query(BaseModel):
#     prompt: str
#     max_length: int = 200
#     temperature: float = 0.7
#     top_p: float = 0.9


# model_id = "Qwen/Qwen2.5-1.5B-Instruct"
# adapter_path = "models/lora"

# try:
#     # Load the base model and tokenizer
#     tokenizer = AutoTokenizer.from_pretrained(model_id)
#     model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")
#     # Load the adapter with PEFT
#     model = PeftModel.from_pretrained(model, adapter_path)

#     # Create a pipeline for text generation
#     gen_pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, return_full_text=False)

# except Exception as e:
#     logger.error(f"Error loading model: {e}")
#     raise


# @app.get("/")
# def read_root():
#     return {"message": "Qwen2.5 Inference API (PEFT loaded)"}


# @app.post("/generate")
# def generate_text(query: Query):
#     try:
#         formatted_prompt = f"Answer this question directly and professionally: {query.prompt}"

#         result = gen_pipe(
#             formatted_prompt,
#             max_new_tokens=query.max_length,
#             temperature=query.temperature,
#             top_p=query.top_p,
#             do_sample=True,
#         )
#         return {"generated_text": result[0]["generated_text"]}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))


# if __name__ == "__main__":
#     import uvicorn

#     uvicorn.run(app, host="0.0.0.0", port=8000)


import argparse

from fastapi import FastAPI, HTTPException
from loguru import logger
from peft import PeftModel  # type: ignore
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

app = FastAPI()


class Query(BaseModel):
    prompt: str
    max_length: int = 200
    temperature: float = 0.7
    top_p: float = 0.9


def create_app(model_id, adapter_path):
    try:
        # Load the base model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")
        # Load the adapter with PEFT
        model = PeftModel.from_pretrained(model, adapter_path)

        # Create a pipeline for text generation
        gen_pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, return_full_text=False)

    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise

    @app.get("/")
    def read_root():
        return {"message": f"Qwen2.5 Inference API (PEFT loaded from {adapter_path})"}

    @app.post("/generate")
    def generate_text(query: Query):
        try:
            formatted_prompt = f"Answer this question directly and professionally: {query.prompt}"

            result = gen_pipe(
                formatted_prompt,
                max_new_tokens=query.max_length,
                temperature=query.temperature,
                top_p=query.top_p,
                do_sample=True,
            )
            return {"generated_text": result[0]["generated_text"]}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    return app


if __name__ == "__main__":
    import uvicorn

    parser = argparse.ArgumentParser(description="Run the inference server")
    parser.add_argument("--model-id", type=str, default="Qwen/Qwen2.5-1.5B-Instruct", help="Model ID to load")
    parser.add_argument("--adapter-path", type=str, default="models/lora", help="Path to the LoRA adapter")
    parser.add_argument("--port", type=int, default=8000, help="Port to run the server on")

    args = parser.parse_args()

    app = create_app(args.model_id, args.adapter_path)

    uvicorn.run(app, host="0.0.0.0", port=args.port)
