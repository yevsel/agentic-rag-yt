from langchain_core.tools import tool
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
import os
from typing import Literal

load_dotenv()

embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

store = PineconeVectorStore(
    index_name=os.environ["INDEX_NAME"],
    embedding=embedding_model
)

@tool
def retrieve_relevant_document(
    query: str,
    blog_post: Literal["harness-engineering","diffusion-models"] | None = None
):
    """
        Semantic search over the two posts.
        
        Pass `blog_post` to search inside one blog post: leave it empty to search both
    """
    if blog_post:
        docs = store.similarity_search(query, k=5, filter={"blog_post": blog_post})
    else:
        docs = store.similarity_search(query, k=5)

    return[
        {
            "blog_post": d.metadata.get("blog_post",""),
            "text": d.page_content
        }
        for d in docs
    ]
    