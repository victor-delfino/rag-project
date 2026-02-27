"""
MCP Server — Expõe o pipeline RAG como ferramentas MCP.

CONCEITO: FastMCP
    O SDK Python do MCP fornece a classe FastMCP, que simplifica
    a criação de servidores MCP. Basta decorar funções com @mcp.tool()
    e elas ficam disponíveis como ferramentas para qualquer cliente MCP.

    Exemplo mínimo:
        mcp = FastMCP("meu-server")

        @mcp.tool()
        def soma(a: int, b: int) -> str:
            return str(a + b)

        mcp.run()  # Inicia servidor via stdio

COMO FUNCIONA O TRANSPORT:
    - stdio: O cliente MCP executa este script como subprocesso
      e se comunica via stdin/stdout. É o modo padrão para uso local
      (Claude Desktop, VS Code, etc.)

    - SSE (Server-Sent Events): O server roda como HTTP server
      e clientes conectam via rede. Útil para deploy remoto.

TOOLS EXPOSTAS:
    1. search_documents — Busca semântica pura (sem LLM)
    2. ask_question — RAG completo (retrieval + LLM)
    3. list_documents — Lista documentos indexados no vector store

Roda no terminal:
    python -m src.mcp_server.server           (stdio - para clientes MCP)
    mcp dev src/mcp_server/server.py          (inspector - para debug)
"""

import sys
from pathlib import Path

# Garante que o projeto raiz está no path para imports funcionarem
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from mcp.server.fastmcp import FastMCP

from src.langchain_rag.chain import create_rag_chain
from src.langchain_rag.retrieval import get_retriever, load_vector_store

# ─── Inicialização do servidor MCP ──────────────────────────────────────────

mcp = FastMCP(
    "rag-project",
    instructions=(
        "Servidor MCP que permite consultar documentos internos da empresa "
        "usando RAG (Retrieval-Augmented Generation). "
        "Possui ferramentas para busca semântica e perguntas com IA."
    ),
)

# ─── Componentes reutilizados (lazy loading) ────────────────────────────────
# Usamos variáveis de módulo para evitar recriar retriever/chain a cada chamada.
# O FastMCP mantém o processo vivo, então só inicializamos uma vez.

_retriever = None
_chain = None


def _get_retriever():
    """Retorna retriever com lazy loading (inicializa apenas na primeira chamada)."""
    global _retriever
    if _retriever is None:
        _retriever = get_retriever(top_k=5)
    return _retriever


def _get_chain():
    """Retorna chain RAG com lazy loading."""
    global _chain
    if _chain is None:
        _chain = create_rag_chain()
    return _chain


# ─── Tool 1: Busca semântica ────────────────────────────────────────────────

@mcp.tool()
def search_documents(query: str, top_k: int = 5) -> str:
    """
    Busca documentos relevantes por similaridade semântica.

    Retorna os trechos mais relevantes dos documentos internos
    SEM gerar uma resposta com IA. Útil para inspecionar quais
    documentos seriam usados para responder uma pergunta.

    Args:
        query: Texto da busca (ex: "política de férias").
        top_k: Número máximo de trechos a retornar (padrão: 5).

    Returns:
        Trechos encontrados formatados com fonte e conteúdo.
    """
    retriever = _get_retriever()

    # Atualiza top_k se diferente do padrão
    retriever.search_kwargs["k"] = top_k

    docs = retriever.invoke(query)

    if not docs:
        return "Nenhum documento encontrado para essa busca."

    results = []
    for i, doc in enumerate(docs, 1):
        source = Path(doc.metadata.get("source", "desconhecido")).name
        results.append(
            f"--- Resultado {i} ---\n"
            f"Fonte: {source}\n"
            f"Conteúdo:\n{doc.page_content}"
        )

    return "\n\n".join(results)


# ─── Tool 2: Pergunta com RAG ───────────────────────────────────────────────

@mcp.tool()
def ask_question(question: str) -> str:
    """
    Faz uma pergunta sobre os documentos internos da empresa.

    Usa RAG (Retrieval-Augmented Generation): primeiro busca
    trechos relevantes nos documentos, depois gera uma resposta
    usando IA com base apenas no conteúdo encontrado.

    O modelo NÃO inventa informações — responde somente com base
    nos documentos indexados.

    Args:
        question: Pergunta em linguagem natural
                  (ex: "Quantos dias de férias eu tenho?").

    Returns:
        Resposta gerada pela IA com base nos documentos encontrados.
    """
    chain = _get_chain()
    return chain.invoke(question)


# ─── Tool 3: Listar documentos ──────────────────────────────────────────────

@mcp.tool()
def list_documents() -> str:
    """
    Lista todos os documentos indexados no sistema.

    Mostra os nomes dos arquivos que foram carregados e podem ser
    consultados. Útil para saber quais informações estão disponíveis.

    Returns:
        Lista de documentos com nomes dos arquivos.
    """
    store = load_vector_store()
    collection = store._collection

    total = collection.count()
    if total == 0:
        return (
            "Nenhum documento indexado. "
            "Execute 'python scripts/ingest.py' para indexar documentos."
        )

    # Busca todos os metadados para extrair nomes dos arquivos
    all_data = collection.get(include=["metadatas"])
    sources = set()
    for meta in all_data["metadatas"]:
        source = meta.get("source", "desconhecido")
        sources.add(Path(source).name)

    lines = [f"Documentos indexados ({len(sources)} arquivos, {total} chunks):\n"]
    for source in sorted(sources):
        lines.append(f"  • {source}")

    return "\n".join(lines)


# ─── Entrypoint ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    mcp.run()
