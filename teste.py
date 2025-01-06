import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage, AIMessage


load_dotenv()

api_key = os.getenv('OPENAI_API_KEY')

if api_key is None:
    raise ValueError("API key not found. Please set the OPENAI_API_KEY environment variable.")
else:
    print("API key loaded successfully.")
    
chat = ChatOpenAI(
    model="gpt-4o-mini",
    max_completion_tokens='1000',
    temperature=0.5 
)

messages = [
    SystemMessage(content="Você é um assistente especializado em segurança cibernética, com foco em análises do OpenVAS. "
                          "Sua função é ajudar a interpretar relatórios, identificar vulnerabilidades e recomendar ações de mitigação."),
    HumanMessage(content="Quais são as vulnerabilidades críticas mais comuns encontradas nos relatórios do OpenVAS?"),
    AIMessage(content="As vulnerabilidades críticas mais comuns incluem falta de patches em sistemas, autenticação fraca, "
                      "exposição de serviços desnecessários, e uso de softwares desatualizados. Você pode mitigá-las "
                      "mantendo os sistemas atualizados, implementando autenticação forte e desativando serviços desnecessários."),
    HumanMessage(content="Como posso priorizar as ações de mitigação?"),
]

res = chat.invoke(messages)

print(res.content)