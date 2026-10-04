import os
import time
from dotenv import load_dotenv


load_dotenv()

from langchain_groq import ChatGroq
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_classic.chains.combine_documents import (
    create_stuff_documents_chain,
)
from langchain_core.prompts import ChatPromptTemplate


class GitHubRepoAnalyzer:
    def __init__(self, repo_path: str):
        self.repo_path = repo_path

        self.llm = ChatGroq(
            model="openai/gpt-oss-120b",
            temperature=0.2,
        )

        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="gemini-embedding-001"
        )

        self.vector_store = None
        self.rag_chain = None

    def load_and_chunk_code(self):
        """
        Load source files from the repository
        and split them into semantic chunks.
        """

        print(f"Loading repository: {self.repo_path}")

        extensions = [
           "**/*.py",
           "**/*.js",
           "**/*.jsx",
           "**/*.ts",
           "**/*.tsx",

           "**/*.java",
           "**/*.cpp",
           "**/*.c",
           "**/*.cs",
           "**/*.go",
           "**/*.rs",

           "**/*.html",
           "**/*.css",

           "**/*.md",
           "**/README.md",

           "**/*package.json",
           "**/*.yaml",
           "**/*.yml",

           "**/*.sql",
           "**/*.toml",

           "**/*.xml",

           "**/*.env.example",

           "**/*.sh",
           "**/*.bat",

           "**/Dockerfile",
           "**/*.dockerfile",
]

        ignore_dirs = {
            ".git",
            ".github",
            ".venv",
            "venv",
            "__pycache__",
            "node_modules",
            "dist",
            "build",
            ".next",
            "coverage",
        }

        documents = []

        for ext in extensions:

            loader = DirectoryLoader(
                self.repo_path,
                glob=ext,
                loader_cls=TextLoader,
                loader_kwargs={"encoding": "utf-8"},
                show_progress=False,
                silent_errors=True,
            )

            docs = loader.load()

            for doc in docs:

                source = doc.metadata.get("source", "")

                if any(folder in source for folder in ignore_dirs):
                    continue

                filename = os.path.basename(source)

                if filename.startswith("test_"):
                    continue

                doc.metadata["filename"] = filename

                doc.page_content = (
                    f"FILE: {filename}\n"
                    f"PATH: {source}\n\n" 
                    f"{doc.page_content}"
                )

                documents.append(doc)
                

        if not documents:
            raise ValueError(
                "No supported source files were found."
            )

        print(f"Loaded {len(documents)} source files.")

        splitter = RecursiveCharacterTextSplitter(
            separators=[
                "\nclass ",
                "\ndef ",
                "\nasync def ",
                "\nfunction ",
                "\nexport ",
                "\nconst ",
                "\nlet ",
                "\n\n",
                "\n",
                " ",
                "",
            ],
            chunk_size=1500,
            chunk_overlap=200,
        )

        chunked_docs = splitter.split_documents(documents)

        print(f"Created {len(chunked_docs)} chunks.")

        return chunked_docs

    def build_vector_database(self, chunked_docs):
        """
        Build an in-memory Chroma vector database in batches.
        This avoids hitting Gemini's free-tier embedding rate limit.
        """

        print("Building vector database...")

        BATCH_SIZE = 20
        WAIT_TIME = 20  # seconds

        self.vector_store = None

        total = len(chunked_docs)

        for i in range(0, total, BATCH_SIZE):

            batch = chunked_docs[i:i + BATCH_SIZE]     #Pass 20 tokens at a time and then take 20 sec break to avoid API hit limit

            print(
                f"Embedding batch {i // BATCH_SIZE + 1} "
                f"({len(batch)} chunks)..."
            )

            if self.vector_store is None:
                import uuid

                self.vector_store = Chroma.from_documents(
                    documents=batch,
                    embedding=self.embeddings,
                    collection_name=f"repo_{uuid.uuid4().hex}",
                )

            else:

                self.vector_store.add_documents(batch)

            processed = min(i + BATCH_SIZE, total)

            print(f"Indexed {processed}/{total} chunks.")

            # Wait before sending the next batch
            if processed < total:

                print(
                    f"Waiting {WAIT_TIME} seconds "
                    "to avoid  rate limits..."
                )

                time.sleep(WAIT_TIME)

        print("Vector database created successfully.")

        self._initialize_chain()

    def _initialize_chain(self):
        """
        Create the Retrieval-Augmented Generation chain.
        """

        retriever = self.vector_store.as_retriever(
            search_kwargs={"k": 10}
        )

        system_prompt ="""You are an expert Software Architect and Senior Software Engineer.

Your job is to explain a GitHub repository to another developer.

Use ONLY the retrieved repository context.

Never invent code or functionality.

If the repository contains README.md, package.json, requirements.txt, pyproject.toml, Dockerfile or similar project description files, use them together with the source code.

When answering:

1. Summarize the project purpose.

2. Mention important folders.

3. Mention important files.

4. Mention important classes.

5. Mention important functions.

6. Explain the backend architecture.

7. Explain the frontend architecture.

8. Explain authentication.

9. Explain database usage.

10. Explain API routes.

11. Explain Socket.IO/WebSocket communication.

12. Explain execution flow from user request to response.

13. Explain how different files interact.

14. Mention important libraries/frameworks.

15. Whenever possible, include exact filenames.

If some information is unavailable, explicitly say:

"The retrieved repository context does not contain this information."

Repository Context:



{context}
"""

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", "{input}"),
            ]
        )

        question_answer_chain = create_stuff_documents_chain(
            self.llm,
            prompt,
        )

        self.rag_chain = create_retrieval_chain(
            retriever,
            question_answer_chain,
        )

    def ask_question(self, question: str):
        """
        Ask a question about the repository.
        """

        if self.rag_chain is None:
            raise ValueError(
                "Repository has not been processed yet."
            )

        response = self.rag_chain.invoke(
            {
                "input": question,
            }
        )

        return response["answer"]