"""
Script de ingestão de documentos.

Roda no terminal: python scripts/ingest.py

O QUE FAZ:
    1. Carrega documentos Markdown do diretório data/
    2. Divide em chunks (RecursiveCharacterTextSplitter)
    3. Gera embeddings + armazena no ChromaDB (via LangChain)
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.langchain_rag.ingestion import load_documents, split_documents
from src.langchain_rag.retrieval import create_vector_store


def main():
    print("=" * 60)
    print("  RAG Project — Ingestão de Documentos")
    print("=" * 60)

    start = time.time()

    # Etapa 1: Carregar documentos
    print("\n📄 Carregando documentos...")
    documents = load_documents()
    print(f"   → {len(documents)} documento(s) carregado(s)")
    for doc in documents:
        source = Path(doc.metadata.get("source", "?")).name
        print(f"     • {source} ({len(doc.page_content)} chars)")

    # Etapa 2: Dividir em chunks
    print("\n🔪 Dividindo em chunks...")
    chunks = split_documents(documents)
    print(f"   → {len(chunks)} chunk(s) gerado(s)")

    # Etapa 3: Embeddar + armazenar (tudo de uma vez!)
    # Chroma.from_documents() gera embeddings e armazena automaticamente.
    print("\n🔢💾 Gerando embeddings e armazenando...")
    vector_store = create_vector_store(chunks)

    elapsed = time.time() - start

    # Estatísticas
    collection = vector_store._collection
    total_stored = collection.count()

    print(f"\n✅ Ingestão concluída em {elapsed:.1f}s!")
    print(f"   Chunks indexados: {total_stored}")
    print("   Agora consulte com: python scripts/ask.py")


if __name__ == "__main__":
    main()
