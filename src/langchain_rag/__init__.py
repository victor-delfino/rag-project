"""
Pacote langchain_rag — Pipeline RAG com LangChain.

ESTRUTURA:
    llm.py         → ChatGroq (modelo de linguagem)
    embeddings.py  → HuggingFaceEmbeddings (vetorização local)
    ingestion.py   → DirectoryLoader + RecursiveCharacterTextSplitter
    retrieval.py   → Chroma vector store + Retriever
    chain.py       → RAG chain (LCEL) + memória de conversa

CONCEITOS CHAVE:
    - LCEL (Expression Language): composição com | (pipe)
    - Chains: sequência declaratíva de passos
    - Retrievers: busca semântica com interface uniforme
    - Memória: histórico de conversa para follow-up contextual
"""
