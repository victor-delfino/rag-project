# RAG Project — Sistema de Perguntas e Respostas com Documentos

Projeto de aprendizado progressivo cobrindo **LLMs**, **RAG**, **LangChain** e **MCP**.

## Objetivo

Construir um sistema que permite fazer perguntas sobre documentos próprios,
usando Retrieval-Augmented Generation (RAG) com LangChain e memória conversacional.

## Stack

- **Python 3.13**
- **Groq Cloud** — LLM (llama-3.3-70b-versatile, gratuito)
- **LangChain** — Orquestração com LCEL (LangChain Expression Language)
- **ChromaDB** — Vector store local (via langchain-chroma)
- **HuggingFace** — Embeddings locais (all-MiniLM-L6-v2, via langchain-huggingface)

## Setup rápido

```bash
# 1. Clone o repositório
git clone https://github.com/<seu-usuario>/rag-project.git
cd rag-project

# 2. Crie o ambiente virtual
python -m venv .venv

# 3. Ative o ambiente
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# 4. Instale as dependências
pip install -e ".[dev]"

# 5. Configure as variáveis de ambiente
cp .env.example .env
# Edite o .env com sua GROQ_API_KEY (obtenha em https://console.groq.com)

# 6. Indexe os documentos
python scripts/ingest.py

# 7. Faça perguntas
python scripts/ask.py
```

## Estrutura do projeto

```
src/
├── config/          → Configurações centralizadas (Settings dataclass)
└── langchain_rag/   → Pipeline RAG com LangChain
    ├── llm.py       → Configuração do LLM (ChatGroq)
    ├── embeddings.py→ Modelo de embeddings local (HuggingFace)
    ├── ingestion.py → Carregamento e chunking de documentos
    ├── retrieval.py → Vector store (ChromaDB) e retriever
    └── chain.py     → Chains LCEL com memória conversacional

scripts/
├── ingest.py        → Indexação de documentos no vector store
└── ask.py           → Chat interativo com RAG + memória

data/                → Documentos para ingestão (Markdown)
tests/               → Testes automatizados
```

## Uso

### Indexar documentos

```bash
python scripts/ingest.py
```

Carrega todos os `.md` de `data/`, divide em chunks e indexa no ChromaDB.

### Chat interativo

```bash
python scripts/ask.py
```

Comandos disponíveis:
- `simple: <pergunta>` — Resposta sem memória
- `debug: <pergunta>` — Mostra chunks recuperados
- `historico` — Exibe histórico da conversa
- `limpar` — Limpa histórico
- `sair` — Encerra

## Fases do projeto

- [x] Fase 1 — Fundamentos conceituais
- [x] Fase 2 — Setup e primeira interação com LLM
- [x] Fase 3 — RAG básico (manual, removido)
- [x] Fase 4 — LangChain em profundidade
- [ ] Fase 5 — MCP (Model Context Protocol)
- [ ] Fase 6 — Evolução e produção
