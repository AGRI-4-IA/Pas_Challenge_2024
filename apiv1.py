from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from unsloth import FastLanguageModel
from fastapi.middleware.cors import CORSMiddleware
import torch

MODEL_NAME = "ibrahima77/agri_llm" 
MAX_SEQ_LENGTH = 2048
LOAD_IN_4BIT = True

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    load_in_4bit=LOAD_IN_4BIT,
)
FastLanguageModel.for_inference(model)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictRequest(BaseModel):
    input_data: str

@app.post("/predict/")
async def predict(request: PredictRequest):
    prompt = f"""
    ### Instruction:
    Recommandation de cultures pour une région donnée au Sénégal !

    ### Input:
    {request.input_data}

    ### Response:
    """
    
    try:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_SEQ_LENGTH).to(device)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erreur lors de la tokenisation : {str(e)}")
    
    try:
        outputs = model.generate(
            inputs["input_ids"], 
            max_new_tokens=128,
            use_cache=True
        )
        raw_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la génération : {str(e)}")
    
    final_response = raw_response.split("### Response:")[-1].strip()
    if not final_response:
        final_response = "Je n'ai pas pu générer une réponse. Pouvez-vous reformuler votre question ?"
    
    return {"response": final_response}
