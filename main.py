import os
import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import google.generativeai as genai
from fastapi.middleware.cors import CORSMiddleware

# --- CONFIGURAÇÃO DA API KEY ---
try:
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("Chave da API do Google não encontrada.")
    genai.configure(api_key=api_key)
except ValueError as e:
    print(e)
    exit()

# --- MODELOS DE DADOS ---
class FeedbackRequest(BaseModel):
    text: str

class FeedbackResponse(BaseModel):
    sentimento: str
    topicos: list[str]
    sumario: str
    insight_acionavel: str

# --- INICIALIZAÇÃO DO FASTAPI ---
app = FastAPI(
    title="InsightStream AI API",
    description="Uma API para analisar feedback de clientes usando IA.",
    version="1.0.0"
)

# --- CONFIGURAÇÃO DO CORS --- <<< ADICIONADO AQUI
origins = [
    "http://localhost:3000",  # O endereço do seu front-end Next.js
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# -----------------------------

# --- LÓGICA DE ANÁLISE ---
def analisar_feedback_com_gemini(texto_do_cliente: str):
    prompt_template = f"""
    Você é um Analista de Produto Sênior especialista em analisar feedback de clientes.
    Analise o seguinte texto e retorne EXATAMENTE um objeto JSON, sem nenhum outro texto ou formatação como markdown.
    A estrutura do JSON deve ser:
    {{
      "sentimento": "positivo" | "negativo" | "neutro",
      "topicos": ["string"],
      "sumario": "string",
      "insight_acionavel": "string"
    }}

    Texto para análise:
    ---
    {texto_do_cliente}
    ---
    """
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        generation_config = genai.types.GenerationConfig(
            response_mime_type="application/json"
        )
        response = model.generate_content(prompt_template, generation_config=generation_config)
        return json.loads(response.text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro na análise da IA: {str(e)}")

# --- ENDPOINTS DA API ---
@app.post("/analyze", response_model=FeedbackResponse)
def analyze_feedback_endpoint(request: FeedbackRequest):
    """
    Recebe um texto de feedback e retorna uma análise de IA estruturada.
    """
    if not request.text or not request.text.strip():
        raise HTTPException(status_code=400, detail="O texto não pode estar vazio.")

    analise = analisar_feedback_com_gemini(request.text)
    return analise

@app.get("/")
def read_root():
    return {"message": "Bem-vindo à API InsightStream AI! Acesse /docs para testar."}