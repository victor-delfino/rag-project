"""
Vector Store e Retriever — Armazenamento e busca semântica.

CONCEITO LANGCHAIN: Retriever
    No LangChain, um Retriever é qualquer objeto que implementa:
        .invoke(query: str) → list[Document]

    O VectorStoreRetriever é o mais comum:
        retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
        docs = retriever.invoke("minha pergunta")

    Mas existem OUTROS tipos de retrievers:
    - MultiQueryRetriever: gera variações da pergunta
    - SelfQueryRetriever: converte a pergunta em filtros
    - ContextualCompressionRetriever: comprime chunks irrelevantes
    - EnsembleRetriever: combina resultados de vários retrievers

    Todos implementam a mesma interface → podem ser trocados sem mudar
    o resto do pipeline.

CONCEITO: from_documents() (class method)
    Chroma.from_documents() cria a store E indexa os documentos
    em uma só chamada:
        store = Chroma.from_documents(docs, embeddings)
"""

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever

from src.config.settings import settings
from src.langchain_rag.embeddings import get_embeddings

# Diretório de persistência do ChromaDB
_PERSIST_DIR = str(settings.project_root / "vector_store")
_COLLECTION_NAME = "rag_documents"


def create_vector_store(documents: list[Document]) -> Chroma:
    """
    Cria um vector store e indexa os documentos.

    O LangChain faz tudo por baixo:
    1. Chama embeddings.embed_documents() em batch
    2. Gera IDs automáticos (UUID)
    3. Faz upsert no ChromaDB
    4. Retorna o store pronto para busca

    IMPORTANTE: Limpa a collection existente antes de recriar,
    evitando duplicatas ao re-indexar.

    Args:
        documents: Lista de Documents (chunks já divididos).

    Returns:
        Instância de Chroma com documentos indexados.
    """
    embeddings = get_embeddings()

    # Limpa collection existente para evitar duplicatas
    existing = Chroma(
        persist_directory=_PERSIST_DIR,
        embedding_function=embeddings,
        collection_name=_COLLECTION_NAME,
    )
    existing.reset_collection()

    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=_PERSIST_DIR,
        collection_name=_COLLECTION_NAME,
    )

    return vector_store


def load_vector_store() -> Chroma:
    """
    Carrega um vector store existente do disco.

    Use quando os documentos JÁ FORAM indexados (por ingest.py).
    Isso evita re-embeddar tudo a cada execução.

    Returns:
        Instância de Chroma conectada ao store existente.
    """
    embeddings = get_embeddings()

    return Chroma(
        persist_directory=_PERSIST_DIR,
        embedding_function=embeddings,
        collection_name=_COLLECTION_NAME,
    )


def get_retriever(top_k: int = 5) -> VectorStoreRetriever:
    """
    Cria um retriever a partir do vector store existente.

    CONCEITO CHAVE: .as_retriever()
        Transforma qualquer VectorStore do LangChain em um Retriever.
        O retriever é o que conecta com as Chains (próximo módulo).

    search_type="similarity":
        Busca por similaridade cosseno (padrão).
        Alternativas:
        - "mmr": Maximal Marginal Relevance (diversifica resultados)
        - "similarity_score_threshold": filtra por score mínimo

    Args:
        top_k: Número de chunks a retornar por busca.

    Returns:
        VectorStoreRetriever pronto para uso em chains.
    """
    store = load_vector_store()

    return store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": top_k},
    )
