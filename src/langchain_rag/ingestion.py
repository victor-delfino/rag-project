"""
Ingestão — Carregamento e chunking de documentos.

CONCEITO LANGCHAIN: Document Loaders
    LangChain tem 700+ loaders prontos:
    - TextLoader: arquivos .txt
    - UnstructuredMarkdownLoader: .md com parsing de headers
    - PyPDFLoader: PDFs
    - CSVLoader: planilhas
    - WebBaseLoader: páginas web
    - GitLoader: repositórios Git
    - NotionDirectoryLoader: exports do Notion

    Cada loader retorna list[Document] com metadados automáticos.

CONCEITO LANGCHAIN: Text Splitters
    RecursiveCharacterTextSplitter é o MAIS USADO:
    - Tenta manter parágrafos inteiros
    - Se não couber, divide por linhas
    - Se ainda não couber, divide por palavras
    - Aplica overlap automaticamente

    separators padrão: ["\\n\\n", "\\n", " ", ""]

NOTA: O Document do LangChain usa `page_content` (não `content`).
"""

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config.settings import settings


def load_documents(directory: str | None = None) -> list[Document]:
    """
    Carrega documentos de um diretório usando LangChain DirectoryLoader.

    COMO O DirectoryLoader FUNCIONA:
        1. Percorre o diretório recursivamente
        2. Para cada arquivo que bate com o glob:
           - Usa o loader_cls para ler o conteúdo
           - Cria um Document com page_content + metadata
        3. Retorna a lista completa

    Args:
        directory: Caminho do diretório. Padrão: settings.data_dir

    Returns:
        Lista de Documents do LangChain.
    """
    data_dir = directory or str(settings.data_dir)

    # DirectoryLoader: varre o diretório e aplica um loader em cada arquivo
    # glob: padrão para filtrar arquivos (ex: "**/*.md")
    # loader_cls: qual loader usar por arquivo (TextLoader para .md e .txt)
    # loader_kwargs: encoding UTF-8 (importante para acentos em português!)
    # show_progress: barra de progresso no terminal
    loader = DirectoryLoader(
        data_dir,
        glob="**/*.md",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        show_progress=True,
    )

    documents = loader.load()
    return documents


def split_documents(
    documents: list[Document],
    chunk_size: int = 800,
    chunk_overlap: int = 200,
) -> list[Document]:
    """
    Divide documentos em chunks usando RecursiveCharacterTextSplitter.

    Args:
        documents: Lista de Documents carregados.
        chunk_size: Tamanho máximo de cada chunk em caracteres.
        chunk_overlap: Sobreposição entre chunks consecutivos.

    Returns:
        Lista de Documents divididos.
    """
    # RecursiveCharacterTextSplitter:
    # separators padrão: ["\\n\\n", "\\n", " ", ""]
    # Para Markdown específico, existe MarkdownTextSplitter.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,           # Conta caracteres (não tokens)
        is_separator_regex=False,      # Separadores são texto literal
        add_start_index=True,          # Adiciona posição do chunk no doc original
    )

    chunks = splitter.split_documents(documents)
    return chunks
