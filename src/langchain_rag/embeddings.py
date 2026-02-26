"""
Embeddings — Modelo de embeddings via LangChain.

CONCEITO LANGCHAIN: Embeddings
    Todos os modelos de embedding no LangChain implementam a interface:
        - embed_query(text: str) → list[float]
        - embed_documents(texts: list[str]) → list[list[float]]

    Note a separação semântica:
    - embed_query: para a PERGUNTA (input do usuário)
    - embed_documents: para os DOCUMENTOS (chunks)

    Em muitos modelos, os dois fazem a mesma coisa. Mas em modelos
    assimétricos (ex: BGE), o embedding da query recebe um prefixo
    diferente do embedding do documento. O LangChain trata isso
    automaticamente.

MODELO ESCOLHIDO: all-MiniLM-L6-v2
    - 384 dimensões (compacto e eficiente)
    - Treinado para similaridade semântica
    - Roda localmente (sem API key, sem custo)
    - Suporta textos de até 256 tokens (~200 palavras)
"""

from langchain_huggingface import HuggingFaceEmbeddings

# Modelo de embedding local
# Se mudarmos o modelo, os embeddings antigos ficam incompatíveis
# e precisaríamos re-indexar todos os documentos.
_DEFAULT_MODEL = "all-MiniLM-L6-v2"


# Cache da instância — evita recarregar o modelo a cada chamada.
# Usamos um dict como cache simples (pattern: memoization por model_name).
_cache: dict[str, HuggingFaceEmbeddings] = {}


def get_embeddings(model_name: str = _DEFAULT_MODEL) -> HuggingFaceEmbeddings:
    """
    Retorna a instância (cacheada) do modelo de embeddings.

    POR QUE HuggingFaceEmbeddings:
        - Gratuito: roda 100% local
        - Privado: seus dados não saem da máquina
        - Modelo: all-MiniLM-L6-v2 (384 dimensões)

    POR QUE CACHE (singleton):
        O modelo leva ~2s para carregar e ocupa ~90MB de RAM.
        Com cache, carrega UMA vez e reutiliza.

    Args:
        model_name: Nome do modelo sentence-transformers.

    Returns:
        Instância de HuggingFaceEmbeddings (reutilizada se já existir).
    """
    if model_name not in _cache:
        _cache[model_name] = HuggingFaceEmbeddings(
            model_name=model_name,
            # model_kwargs: parâmetros passados ao SentenceTransformer
            model_kwargs={"device": "cpu"},
            # encode_kwargs: parâmetros passados ao .encode()
            encode_kwargs={"normalize_embeddings": True},
        )
    return _cache[model_name]
