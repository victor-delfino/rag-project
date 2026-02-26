"""
LLM — Configuração do modelo de linguagem via LangChain.

CONCEITO LANGCHAIN: BaseChatModel
    Todos os LLMs no LangChain herdam de BaseChatModel.
    Isso significa que ChatGroq, ChatOpenAI, ChatAnthropic implementam
    a MESMA interface: .invoke(), .stream(), .batch().

    Para trocar de provedor:
        llm = ChatGroq(model="llama-3.3-70b-versatile")
        llm = ChatOpenAI(model="gpt-4o-mini")
        llm = ChatAnthropic(model="claude-3-haiku")

    O resto do código NÃO MUDA. Esse é o poder da abstração.
    LangChain usa herança/polimorfismo (OOP clássico).
"""

from langchain_groq import ChatGroq

from src.config.settings import settings


def get_llm(temperature: float = 0.3) -> ChatGroq:
    """
    Cria e retorna uma instância do LLM configurado.

    POR QUE UMA FUNÇÃO FACTORY (e não uma variável global):
        - Podemos passar parâmetros diferentes cada vez (temperature)
        - Mais fácil de testar (não depende de import-time side effects)
        - Padrão comum em LangChain: criar na hora, configurar por uso

    PARÂMETROS IMPORTANTES:
        temperature: controla a "criatividade" do modelo
            - 0.0: determinístico (mesma pergunta → mesma resposta)
            - 0.3: levemente variado (bom para RAG factual)
            - 0.7-1.0: criativo (bom para brainstorming, histórias)

        max_tokens: limite de tokens na resposta
            - None: usa o padrão do modelo
            - 1024: respostas concisas
            - 4096: respostas longas

    Args:
        temperature: Grau de aleatoriedade das respostas (0.0 a 1.0).

    Returns:
        Instância de ChatGroq pronta para uso.
    """
    if not settings.groq_api_key:
        raise ValueError(
            "GROQ_API_KEY não configurada no .env.\n"
            "Obtenha em: https://console.groq.com/keys"
        )

    return ChatGroq(
        model=settings.groq_model,
        api_key=settings.groq_api_key,
        temperature=temperature,
        max_tokens=1024,
        # max_retries já é tratado internamente pelo LangChain (padrão: 2)
    )
