from langchain_ollama import ChatOllama
from dotenv import load_dotenv

load_dotenv()

# Local model: free, no API key, no credits.
#
# num_ctx matters. Ollama gives models a small context window by default, and
# our prompts carry up to 10 chunks, so without this the prompt gets silently
# cut off and answers turn to nonsense.
llm = ChatOllama(
    model="llama3.1:8b",
    temperature=0,
    num_ctx=8192,
)
