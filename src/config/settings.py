"""
Configurações centralizadas do projeto.

POR QUE ESTE ARQUIVO EXISTE:
    Em vez de espalhar `os.getenv("GROQ_API_KEY")` por todo o código,
    centralizamos tudo aqui. Isso significa que:
    - Se uma variável muda de nome, você altera em UM lugar
    - Fica claro quais configurações o projeto usa
    - Podemos validar configurações na inicialização (fail fast)

POR QUE USAR dataclass:
    - Simples, nativa do Python (sem dependência extra)
    - Tipada (IDE mostra autocomplete)
    - Imutável com frozen=True (evita alterações acidentais)
"""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# Carrega o .env da raiz do projeto
# Path(__file__) = este arquivo (settings.py)
# .parent.parent.parent = sobe 3 níveis (config → src → rag-project)
_PROJECT_ROOT = Path(__file__).parent.parent.parent
load_dotenv(_PROJECT_ROOT / ".env")


@dataclass(frozen=True)
class Settings:
    """Configurações imutáveis do projeto."""

    # Groq (cloud gratuito — LLM via LangChain)
    # Obtenha sua API key em: https://console.groq.com/keys
    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    groq_model: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

    # Caminhos do projeto
    project_root: Path = _PROJECT_ROOT
    data_dir: Path = _PROJECT_ROOT / "data"


# Instância única de configuração (Singleton simples)
# Importar assim: from src.config.settings import settings
settings = Settings()
