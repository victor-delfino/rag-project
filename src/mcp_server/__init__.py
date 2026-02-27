"""
Pacote mcp_server — Servidor MCP (Model Context Protocol).

O QUE É MCP:
    O Model Context Protocol (MCP) é um protocolo aberto que padroniza
    a comunicação entre LLMs e fontes de dados/ferramentas externas.

    Pense assim:
    - Sem MCP: cada app LLM implementa suas próprias integrações
    - Com MCP: qualquer cliente MCP pode se conectar a qualquer servidor MCP

    É como USB para LLMs — um padrão universal de conexão.

ARQUITETURA:
    ┌──────────────┐     stdio/SSE     ┌──────────────┐
    │ Cliente MCP  │ ◄───────────────► │ Servidor MCP │
    │ (Claude, VS  │                   │ (este código)│
    │  Code, etc.) │                   │              │
    └──────────────┘                   │  Tools:      │
                                       │  - search    │
                                       │  - ask       │
                                       │  - list_docs │
                                       └──────────────┘

CONCEITOS CHAVE:
    - Tools: Funções que o LLM pode chamar (como function calling)
    - Resources: Dados que o LLM pode ler (como arquivos)
    - Prompts: Templates reutilizáveis para o LLM
    - Transport: stdio (local) ou SSE (rede)

    Neste projeto usamos apenas Tools, que é o caso de uso mais comum.
"""
