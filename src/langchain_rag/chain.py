"""
RAG Chain — Pipeline completo de Retrieval-Augmented Generation.

O QUE É UMA CHAIN:
    Uma Chain é uma SEQUÊNCIA DE PASSOS conectados, onde a saída de um
    passo alimenta o próximo. No RAG:

    Pergunta → Retriever → Contexto → Prompt → LLM → Resposta

LCEL — LangChain Expression Language:
    O LangChain usa o operador | (pipe) para compor chains:

        chain = retriever | prompt | llm | parser

    Isso é inspirado em pipes Unix: cat file | grep "erro" | wc -l
    Cada componente recebe input, processa e passa para o próximo.

    INTERNAMENTE, cada | cria um RunnableSequence que pode:
    - Ser invocado: chain.invoke("pergunta")
    - Ser streamado: chain.stream("pergunta")
    - Ser executado em batch: chain.batch(["p1", "p2", "p3"])
    - Ser executado em async: await chain.ainvoke("pergunta")

MEMÓRIA DE CONVERSA:
    Com memória, o assistente lembra das perguntas anteriores:

    User: Quais são os benefícios?
    AI: Plano de saúde, vale-refeição, Gympass...
    User: E o plano de saúde, como funciona?     ← refere-se ao anterior!
    AI: O plano de saúde cobre...                ← responde com contexto!

    Implementamos com um chat_history manual que é passado como
    variável ao prompt template.
"""

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough

from src.langchain_rag.llm import get_llm
from src.langchain_rag.retrieval import get_retriever


def _format_docs(docs: list[Document]) -> str:
    """
    Formata documentos recuperados em texto para o prompt.

    Formato: [Fonte: arquivo.md] seguido do conteúdo.
    """
    parts = []
    for doc in docs:
        source = doc.metadata.get("source", "desconhecido")
        parts.append(f"[Fonte: {source}]\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


# ─── CHAIN SIMPLES (sem memória) ─────────────────────────────────────────────

def create_rag_chain():
    """
    Cria o pipeline RAG completo usando LCEL (LangChain Expression Language).

    ANATOMIA DA CHAIN:
        A chain é lida da esquerda para a direita:

        1. {"context": retriever | format, "question": passthrough}
           → Busca chunks E repassa a pergunta intacta

        2. | prompt
           → Injeta context + question no template

        3. | llm
           → Envia o prompt para o Groq

        4. | StrOutputParser()
           → Extrai o texto da resposta (AIMessage → str)

    O TRUQUE DO RunnablePassthrough:
        Precisamos que a pergunta siga DOIS caminhos ao mesmo tempo:
        - Para o retriever (buscar chunks)
        - Para o prompt (como a pergunta no template)

        RunnablePassthrough() simplesmente "passa adiante" o input
        sem alterá-lo. É como um fio que conecta a entrada direto
        ao prompt.

    Returns:
        Chain invocável: chain.invoke("minha pergunta") → str
    """
    retriever = get_retriever()
    llm = get_llm(temperature=0.3)

    # Template do prompt — equivalente ao _RAG_PROMPT_TEMPLATE da Fase 3
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "Você é um assistente que responde perguntas com base em "
            "documentos internos da empresa. Responda em português "
            "de forma clara e objetiva.",
        ),
        (
            "human",
            "Responda a pergunta abaixo APENAS com base no contexto fornecido.\n"
            "Se a resposta não puder ser encontrada no contexto, diga:\n"
            '"Não encontrei essa informação nos documentos disponíveis."\n'
            "Não invente informações. Cite o documento de origem quando possível.\n\n"
            "CONTEXTO:\n{context}\n\n"
            "PERGUNTA: {question}\n\n"
            "RESPOSTA:",
        ),
    ])

    # LCEL: composição com o operador |
    # Leia assim: "o input vai para o retriever E para o passthrough,
    # depois o resultado vai para o prompt, depois para o LLM,
    # depois para o parser"
    chain = (
        {
            "context": retriever | _format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain


# ─── CHAIN COM MEMÓRIA ───────────────────────────────────────────────────────

def create_conversational_rag_chain():
    """
    Cria um pipeline RAG com memória de conversa.

    CONCEITO NOVO: MEMÓRIA
        Na Fase 3, cada pergunta era independente.
        Agora, o assistente LEMBRA das perguntas anteriores.

    COMO FUNCIONA:
        1. O chat_history é passado como variável no prompt
        2. O modelo recebe: system + histórico + pergunta atual
        3. O modelo usa o histórico para entender contexto
           Ex: "E sobre o vale-refeição?" → sabe que estamos falando de benefícios

    POR QUE NÃO REFORMULAR A QUERY:
        Uma abordagem mais sofisticada seria reformular a pergunta antes
        da busca. Ex: "E sobre o vale-refeição?" → "Quais são as regras
        do vale-refeição da empresa?". Isso é o create_history_aware_retriever.
        Vamos mantê-lo simples aqui e evoluir na Fase 6.

    Returns:
        Tuple de (chain, chat_history):
        - chain: invocável com chain.invoke({"question": ..., "chat_history": ...})
        - chat_history: lista mutável para acumular mensagens
    """
    retriever = get_retriever()
    llm = get_llm(temperature=0.3)

    # Template com placeholder para o histórico de conversa
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "Você é um assistente que responde perguntas com base em "
            "documentos internos da empresa. Responda em português "
            "de forma clara e objetiva. "
            "Use o histórico da conversa para entender o contexto "
            "das perguntas do usuário.",
        ),
        # MessagesPlaceholder: insere a lista de mensagens do histórico aqui
        # Isso mantém o formato correto (HumanMessage, AIMessage alternando)
        MessagesPlaceholder(variable_name="chat_history"),
        (
            "human",
            "Responda a pergunta abaixo APENAS com base no contexto fornecido.\n"
            "Se a resposta não puder ser encontrada no contexto, diga:\n"
            '"Não encontrei essa informação nos documentos disponíveis."\n'
            "Não invente informações. Cite o documento de origem quando possível.\n\n"
            "CONTEXTO:\n{context}\n\n"
            "PERGUNTA: {question}\n\n"
            "RESPOSTA:",
        ),
    ])

    chain = (
        {
            "context": (lambda x: x["question"]) | retriever | _format_docs,
            "question": lambda x: x["question"],
            "chat_history": lambda x: x["chat_history"],
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    # Histórico vazio — será preenchido pelo script de uso
    chat_history: list[HumanMessage | AIMessage] = []

    return chain, chat_history
