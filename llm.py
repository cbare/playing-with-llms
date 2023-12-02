import os
from  langchain.llms import CTransformers
from langchain import PromptTemplate, LLMChain

local_llm = "zephyr-7b-beta.Q5_K_M.gguf"

config = {
    "max_new_tokens": 4096,
    "repetition_penality": 1.1,
    "temperature": 0.5,
    "top_k": 50,
    "top_p": 0.9,
    "stream": True,
    "threads": int(os.cpu_count()/2)
}

model = CTransformers(
    model = local_llm,
    model_type = "mistral",
    lib = "avx2",
    **config,
)

print(model)

query = "Who played piano with jazz saxophone legend John Coltrane?"
for text in model(query):
    print(text, end="", flush=True)


query = ""
for text in model(query):
    print(text, end="", flush=True)

hist = ""
query = ""
hist += "user: " + query + "\n\nassistant: "
for text in model(query):
    print(text, end="", flush=True)
    hist += text
hist += "\n\n"