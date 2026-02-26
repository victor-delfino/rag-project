"""
Script de perguntas e respostas — COM MEMÓRIA DE CONVERSA.

Roda no terminal: python scripts/ask.py

MODOS DE USO:
    - Modo normal: faz pergunta, recebe resposta com fontes
    - Modo simples: prefixar com "simple:" para chain sem memória
    - Modo debug: prefixar com "debug:" para ver chunks recuperados
    - Comando "historico": mostra o histórico da conversa
    - Comando "limpar": limpa a memória

EXPERIMENTE A MEMÓRIA:
    ❓ Quais são os benefícios da empresa?
    📝 Plano de saúde, vale-refeição, Gympass...
    ❓ E como funciona o plano de saúde?     ← O "o plano" refere ao anterior!
    📝 O plano de saúde cobre...             ← Funciona por causa da memória
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_core.messages import AIMessage, HumanMessage

from src.langchain_rag.chain import create_conversational_rag_chain, create_rag_chain
from src.langchain_rag.retrieval import get_retriever, load_vector_store


def main():
    print("=" * 60)
    print("  RAG Project — Q&A com Memória de Conversa")
    print("=" * 60)

    # Verificar se há documentos indexados
    try:
        store = load_vector_store()
        count = store._collection.count()
    except Exception:
        count = 0

    if count == 0:
        print(
            "\n❌ Vector store vazio! Rode primeiro:\n"
            "   python scripts/ingest.py\n"
        )
        sys.exit(1)

    print(f"\n📚 Vector store: {count} chunks indexados")

    # Criar chain COM memória (principal)
    print("🔗 Criando chain RAG conversacional...")
    conv_chain, chat_history = create_conversational_rag_chain()

    # Criar chain simples (sem memória, para comparação)
    simple_chain = create_rag_chain()

    # Retriever para modo debug
    retriever = get_retriever()

    print("\n💬 Chat com memória de conversa ativo!")
    print("   Comandos: 'historico', 'limpar', 'sair'")
    print("   Prefixos: 'debug:' (ver chunks), 'simple:' (sem memória)\n")

    while True:
        try:
            question = input("❓ Pergunta: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\nEncerrando...")
            break

        if not question:
            continue

        lower = question.lower()

        if lower in ("sair", "exit", "quit"):
            print("\nAté mais! 👋")
            break

        # Comando: mostrar histórico
        if lower == "historico":
            if not chat_history:
                print("\n📝 Histórico vazio.\n")
            else:
                print(f"\n📝 Histórico ({len(chat_history)} mensagens):")
                for msg in chat_history:
                    role = "👤" if isinstance(msg, HumanMessage) else "🤖"
                    preview = msg.content[:100]
                    suffix = "..." if len(msg.content) > 100 else ""
                    print(f"   {role} {preview}{suffix}")
                print()
            continue

        # Comando: limpar memória
        if lower == "limpar":
            chat_history.clear()
            print("\n🧹 Memória limpa!\n")
            continue

        # Modo debug: ver chunks recuperados
        if lower.startswith("debug:"):
            query = question[6:].strip()
            print("\n🔍 Buscando chunks relevantes...")
            try:
                docs = retriever.invoke(query)
                print(f"   → {len(docs)} chunk(s):\n")
                for i, doc in enumerate(docs):
                    source = Path(doc.metadata.get("source", "?")).name
                    preview = doc.page_content[:150].replace("\n", " ")
                    print(f"   [{i + 1}] {source}")
                    print(f"       {preview}...\n")
            except Exception as e:
                print(f"\n❌ Erro: {e}")
            print()
            continue

        # Modo simples: chain SEM memória (comparação com Fase 3)
        if lower.startswith("simple:"):
            query = question[7:].strip()
            print("\n🔍 [Chain simples — sem memória]")
            try:
                answer = simple_chain.invoke(query)
                print(f"\n📝 Resposta:\n{answer}")
            except Exception as e:
                print(f"\n❌ Erro: {e}")
            print()
            continue

        # Modo padrão: chain COM memória
        try:
            print("\n🔍 Buscando nos documentos (com contexto da conversa)...", flush=True)

            answer = conv_chain.invoke({
                "question": question,
                "chat_history": chat_history,
            })

            # Adicionar ao histórico
            chat_history.append(HumanMessage(content=question))
            chat_history.append(AIMessage(content=answer))

            print(f"\n📝 Resposta:\n{answer}")
            print(f"\n   💭 Memória: {len(chat_history) // 2} turno(s) no histórico")

        except Exception as e:
            print(f"\n❌ Erro: {e}")

        print()


if __name__ == "__main__":
    main()
