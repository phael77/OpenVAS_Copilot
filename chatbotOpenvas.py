#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI


# In[2]:


import os
from dotenv import load_dotenv

# Carregar variáveis de ambiente do arquivo .env
load_dotenv()

api_key = os.getenv('OPENAI_API_KEY')

if api_key is None:
    raise ValueError("API key not found. Please set the OPENAI_API_KEY environment variable.")
else:
    print("API key loaded successfully.")


# In[4]:


chat = ChatOpenAI(
    api_key=api_key,
    model="gpt-4o-mini",
    max_completion_tokens=10000,
)


# In[5]:


from langchain.schema import SystemMessage, HumanMessage, AIMessage

messages = [
    SystemMessage(content="Você é um assistente especializado em segurança cibernética, com foco em análises do OpenVAS. "
                          "Sua função é ajudar a interpretar relatórios, identificar vulnerabilidades e recomendar ações de mitigação."),
    HumanMessage(content="Quais são as vulnerabilidades críticas mais comuns encontradas nos relatórios do OpenVAS?"),
    AIMessage(content="As vulnerabilidades críticas mais comuns incluem falta de patches em sistemas, autenticação fraca, "
                      "exposição de serviços desnecessários, e uso de softwares desatualizados. Você pode mitigá-las "
                      "mantendo os sistemas atualizados, implementando autenticação forte e desativando serviços desnecessários."),
    HumanMessage(content="Como posso priorizar as ações de mitigação?"),
]


# In[6]:


res = chat.invoke(messages)


# In[7]:


res


# In[8]:


print(res.content)


# In[9]:


messages.append(res)


# In[10]:


messages


# In[11]:


prompt = HumanMessage(content="Você poderia me explicar como funciona o OpenVAS?")
messages.append(prompt)


# In[12]:


res = chat.invoke(messages)
print(res.content)


# In[13]:


import pandas as pd
df = pd.read_csv('data/miniopenvas.csv')


# In[14]:


context = [
    "Aqui está um exemplo de relatório do OpenVAS:",
    df.to_string(index=False),
    "Quero que você me ajude a interpretar este relatório e identificar as vulnerabilidades críticas."
]

source_knowledge = "\n".join(context)


# In[15]:


query = "Quero que me ajude a interpretar este relatório e identificar as vulnerabilidades mais críticas e sugerir a priorizar a mitigação das vulnerabilidades mais critivas."

augmented_prompt = f"""
    Use o contexto abaixo para responder à seguinte pergunta:

    Contexto: {source_knowledge}

    Pergunta: {query}

    Boa prática: não forneça o endereço IP ou nome do host do sistema nas respostas
"""


# In[16]:


prompt = HumanMessage(content=augmented_prompt)

messages.append(prompt)

res = chat.invoke(messages, )
print(res.content)


# Embeddings

# In[17]:


data = pd.read_csv('data/openvas.csv')


# In[18]:


from langchain_openai import OpenAIEmbeddings
embed_model = OpenAIEmbeddings(model="text-embedding-3-small")


# In[19]:


texts = [
    'this is a test',
    'this is another test',
]

res = embed_model.embed_documents(texts)
len(res), len(res[0])


# In[20]:


docs = data[[
    'Port', 'Port Protocol', 'CVSS', 'Severity', 'QoD', 'Solution Type', 'NVT Name', 'Summary', 'Specific Result', 'NVT OID',
    'CVEs', 'Impact', 'Solution', 'Affected Software/OS', 'Vulnerability Insight', 'Vulnerability Detection Method', 'Product Detection Result',
    ]]
docs.head()


# In[21]:


from langchain_community.document_loaders import DataFrameLoader

loader = DataFrameLoader(docs, page_content_column='Summary')
documents = loader.load()


# In[22]:


documents[99]


# In[23]:


documents[99].page_content


# In[26]:


documents[999].metadata


# In[27]:


from langchain_community.vectorstores import Qdrant
from langchain_openai import OpenAIEmbeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

qdrant = Qdrant.from_documents(
    documents = documents, 
    embedding = embeddings,
    location = ":memory:",
    collection_name = "chatbot"


)


# In[32]:


query = "Quais são as principais vulnerabilidaes encontradas no relatório? E suas soluções?"

qdrant.similarity_search(query, k=100)


# In[34]:


def custom_prompt(query: str):
    results = qdrant.similarity_search(query, k=100)
    source_knowledge = "\n".join(x.page_content for x in results)
    augmented_prompt = f"""
        Use o contexto abaixo para responder à seguinte pergunta:

        Contexto: {source_knowledge}

        Pergunta: {query}

        Boa prática: não forneça o endereço IP ou nome do host do sistema nas respostas

        Seja claro nas respostas e forneça informações relevantes para ajudar o usuário a entender o conteúdo. Lembre-se de que o objetivo é ajudar a interpretar relatórios do OpenVAS e identificar vulnerabilidades críticas.

        Os usuários podem ter diferentes níveis de conhecimento em segurança cibernética, então tente explicar os conceitos de forma simples e objetiva.
    """
    return augmented_prompt


# In[35]:


print(custom_prompt(query))


# In[36]:


prompt = HumanMessage(content=custom_prompt(query))

messages.append(prompt)

res = chat.invoke(messages)

print(res.content)


# In[37]:


teste = pd.read_csv('data/teste.csv')


# In[38]:


prompt_teste = teste.to_string(index=False)


# In[39]:


prompt = HumanMessage(content=f"""
    Use o contexto abaixo para responder à seguinte pergunta:
                         
    Contexto: {prompt_teste}

    Pergunta: Me ajude a resolver os problemas encontrados nesse relatório.
""")

messages.append(prompt)

res = chat.invoke(messages)

print(res.content)

