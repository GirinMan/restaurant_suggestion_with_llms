import torch
import copy
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import List
import requests
import json

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/polyglot-ko-5.8b")

model = AutoModelForCausalLM.from_pretrained(
    "EleutherAI/polyglot-ko-5.8b",
    pad_token_id=tokenizer.eos_token_id,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    device_map="auto",
    load_in_8bit=False,
)
model.eval()

peft_model_id = "../../template_based_SFT/checkpoints/polyglot-ko-5.8b-lora-parse_cmd/2023-05-15_14:30:24"
model = PeftModel.from_pretrained(model, peft_model_id, adapter_name="latest")

ckpts = {f"step-{i}" : peft_model_id + f"/checkpoint-{i}" for i in [15*j for j in range(1, 11)]}
for name, path in ckpts.items():
    model.load_adapter(path, adapter_name=name)
    print(name)

def generate(text, name):
    model.set_adapter(name)
    tokenized_text = tokenizer.encode(text, return_tensors='pt').to('cuda')
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=tokenized_text,
            eos_token_id = tokenizer.eos_token_id,
            do_sample=False, #샘플링 전략 사용
            num_beams=3,
            max_new_tokens= 64, # 최대 디코딩 길이
            top_k=0, # 확률 순위 밖인 토큰은 샘플링에서 제외
            top_p=0.8, # 누적 확률 이내의 후보집합에서만 생성
            temperature=0.19,
            no_repeat_ngram_size = 4,
        )
    #torch.cuda.empty_cache()
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return generated_text

app = FastAPI(title='kogpt',
              description='kakaobrain-kogpt6B-fp16',
              version="1.0")


class textInput(BaseModel):
    """Input model for prediction
    """
    text: str = Field(None, description='Prompt', example="여기에 입력")
    adapter_name: str = Field(None, description='LoRA to use for generate', example="step-8")


class textResponse(BaseModel):
    text: str = Field(None, description="generated texts")


@app.get("/")
def home():
    return "Refer to '/docs' for API documentation"


@app.post("/generate", description="Generation", response_model=textResponse)
def get_generation(req_body: textInput):
    """Prediction
    :param req_body:
    :return:
    """
    #torch.cuda.empty_cache()
    result = generate(req_body.text, req_body.adapter_name)
    return {"text":result}
