# 🎓 Beginner-Friendly SDE Interview Preparation Guide

## Project: GitHub Repository Analyzer (GenAI / RAG)

This README explains the project in simple language. It is designed for someone who is new to Retrieval-Augmented Generation (RAG) and wants to understand both the code and the design decisions.

> **Important:** This document describes the implementation shown in the project files. Performance figures such as response time, number of users, or onboarding-time reduction should only be claimed if you have actually measured them. The project code uses `openai/gpt-oss-120b` through Groq; if another model is named in an older README, check the code before describing the current setup.

---

# TABLE OF CONTENTS

1. [PART 1: Project Deep Dive](#part-1-project-deep-dive)
   - [1.1 Short and Detailed Project Introductions](#11-short-and-detailed-project-introductions)
   - [1.2 Problem Statement and Motivation](#12-problem-statement-and-motivation)
   - [1.3 System Architecture and Workflow](#13-system-architecture-and-workflow)
   - [1.4 Component-by-Component Walkthrough](#14-component-by-component-walkthrough)
   - [1.5 Engineering Challenges and Solutions](#15-engineering-challenges-and-solutions)
   - [1.6 Explaining Metrics Honestly](#16-explaining-metrics-honestly)
2. [PART 2: Technology Choices and Code Details](#part-2-technology-choices-and-code-details)
   - [2.1 Why These Technologies?](#21-why-these-technologies)
   - [2.2 Important Functions and Parameters](#22-important-functions-and-parameters)
3. [PART 3: 50 Interview Questions and Answers](#part-3-50-interview-questions-and-answers)
   - [Category A: Project Overview (Q1–Q8)](#category-a-project-overview-q1q8)
   - [Category B: Architecture and Design (Q9–Q18)](#category-b-architecture-and-design-q9q18)
   - [Category C: Code and LangChain (Q19–Q28)](#category-c-code-and-langchain-q19q28)
   - [Category D: RAG and Vector Databases (Q29–Q38)](#category-d-rag-and-vector-databases-q29q38)
   - [Category E: Errors and Security (Q39–Q44)](#category-e-errors-and-security-q39q44)
   - [Category F: Scaling and Production (Q45–Q50)](#category-f-scaling-and-production-q45q50)
4. [Quick Interview Cheat Sheet](#quick-interview-cheat-sheet)

---

# PART 1: Project Deep Dive

## 1.1 Short and Detailed Project Introductions

### ⏱️ 30-second introduction

> “I built a GenAI-based GitHub Repository Analyzer using Retrieval-Augmented Generation, or RAG. A user enters a public GitHub repository URL, and the application clones the repository, reads supported code and text files, splits their content into smaller chunks, and stores their embeddings in ChromaDB. The user can then ask questions about the codebase in natural language. The system retrieves relevant code chunks and sends them to an LLM through Groq to generate an answer based on the repository context.”

### ⏱️ Two-minute introduction

> “Understanding an unfamiliar codebase can take time because a developer may need to open many files, trace functions, and understand how different components work together. This project helps make that process easier.
>
> The application has two main stages: **indexing** and **question answering**.
>
> During indexing, the user provides a public GitHub URL. GitPython clones the repository into a temporary local folder. LangChain loaders read supported source-code and configuration files. A text splitter divides the files into smaller chunks, while file names and paths are added to the content to help identify where each chunk came from.
>
> The application uses Gemini Embeddings to convert the chunks into numerical vectors. These vectors, along with their text and metadata, are stored in ChromaDB.
>
> When the user asks a question, the retriever searches ChromaDB for the ten most relevant chunks. LangChain places those chunks and the question into a prompt, and the Groq-hosted LLM generates the response. Streamlit displays the answer in a chat interface.
>
> The main idea is that we do not send the entire repository to the LLM for every question. We first retrieve the parts that appear relevant and then use them as context for the answer.”

### A simple way to remember the project

- **GitPython:** downloads the repository.
- **DirectoryLoader and TextLoader:** read supported files.
- **Text splitter:** divides long file contents into chunks.
- **Gemini Embeddings:** turns chunks into vectors.
- **ChromaDB:** stores vectors and their associated documents.
- **Retriever:** finds chunks related to a question.
- **Groq LLM:** writes the answer using the retrieved context.
- **Streamlit:** provides the user interface.

---

## 1.2 Problem Statement and Motivation

### The problem

When a developer opens an unfamiliar repository, they may need to find answers to questions such as:

- Where is authentication implemented?
- Which file defines the API routes?
- How does the application connect to its database?
- Which function handles a particular operation?
- How does a request move from the frontend to the backend?

Manually answering these questions may require searching through many files. Documentation may also be incomplete or outdated.

### Why not simply send the whole repository to an LLM?

A large repository can contain too much text to send in one request. Even when a model supports a large context window:

1. Sending large amounts of code can increase token usage, latency, and cost.
2. The important code may be surrounded by a lot of unrelated content.
3. The model may still misunderstand the code or make unsupported claims.
4. The repository may change, so the system needs a way to index the current version.

### The solution: RAG

RAG stands for **Retrieval-Augmented Generation**.

In simple terms:

1. Store information from the repository in a searchable knowledge base.
2. When a user asks a question, search for the most relevant pieces of information.
3. Give those pieces to the LLM along with the question.
4. Ask the LLM to generate an answer using that context.

RAG does not retrain the LLM every time a new repository is uploaded. Instead, the application builds or updates its knowledge base.

### Important distinction

RAG can help an LLM answer questions about specific repository content, but it does **not** guarantee that every answer is correct. The retriever may miss useful code, and the LLM may still misunderstand the context. Retrieval quality and answer quality should be tested.

---

## 1.3 System Architecture and Workflow

### High-level workflow

```text
                    USER
                      |
                      | Enters public GitHub URL
                      v
              Streamlit Interface
                   (app.py)
                      |
                      v
             GitPython clone helper
                (github_utils.py)
                      |
                      v
             Temporary local folder
                      |
                      v
             GitHubRepoAnalyzer
               (repo_analyzer.py)
                      |
                      v
        DirectoryLoader + TextLoader
                      |
                      v
              Source Documents
                      |
                      v
          RecursiveCharacterTextSplitter
              chunks of ~1500 characters
                 with 200 overlap
                      |
                      v
              Gemini Embeddings
                      |
                      v
             ChromaDB Vector Store
                      |
            INDEXING STAGE COMPLETE
                      |
                      v
               User asks a question
                      |
                      v
                RAG retrieval chain
                      |
                      v
          Retrieve top 10 relevant chunks
                      |
                      v
             Question + code context
                      |
                      v
               Groq-hosted LLM
                      |
                      v
             Answer shown in Streamlit
```

### Stage 1: Indexing

Indexing happens after the user clicks **Analyze**.

1. The app checks that the repository URL and required API key are present.
2. The repository is cloned to a temporary directory.
3. Supported files are loaded as LangChain `Document` objects.
4. Unwanted paths and some test files are filtered out.
5. File names and paths are added to document content.
6. Each document is split into smaller chunks.
7. Gemini creates an embedding for each chunk.
8. ChromaDB stores the chunks and their vector representations.

Indexing may take time because a repository can contain many files and embedding calls may be rate-limited.

### Stage 2: Question answering

Question answering happens after the repository has been indexed.

1. The user enters a question in the Streamlit chat box.
2. `ask_question()` passes it to the RAG chain.
3. The retriever uses the embedding model and ChromaDB to find relevant chunks.
4. The retrieved chunks are inserted into the prompt's context.
5. The prompt and question are sent to the LLM through Groq.
6. The LLM generates an answer.
7. Streamlit displays the answer and adds it to the chat history.

### Indexing time vs inference time

- **Indexing time:** time spent cloning, reading, chunking, embedding, and storing repository content.
- **Inference/query time:** time spent processing a question, retrieving relevant chunks, and generating the answer.

These are different stages. Do not describe repository cloning and initial indexing as part of every question's inference time.

---

## 1.4 Component-by-Component Walkthrough

### 1. `github_utils.py` — Cloning and cleanup

This file contains two helper functions.

#### `clone_repository(repo_url)`

Its job is to create a local copy of the public GitHub repository.

The function:

1. Removes a trailing slash from the URL.
2. Adds `.git` if the URL does not already end with it.
3. Creates a temporary directory using `tempfile.mkdtemp(prefix="repo_")`.
4. Uses `Repo.clone_from()` from GitPython to clone the repository into that directory.
5. Returns the temporary directory path.

Example:

```python
temp_dir = tempfile.mkdtemp(prefix="repo_")
Repo.clone_from(repo_url, temp_dir)
return temp_dir
```

On Windows, the temporary folder is usually created somewhere under the user's local temporary directory, for example:

```text
C:\Users\<username>\AppData\Local\Temp\repo_...
```

The exact path depends on the operating system and its temporary-directory configuration.

#### `delete_repository(repo_path)`

This function removes the cloned repository from disk:

```python
shutil.rmtree(repo_path, ignore_errors=True)
```

It is called when the user clears the current analysis and when certain processing errors occur. It helps prevent temporary repository copies from accumulating.

**Interview answer:**

> “GitPython lets the application work with Git repositories from Python code. I use it to clone a public repository into a temporary folder, analyze its files, and remove that folder when cleanup is needed.”

### 2. `repo_analyzer.py` — The RAG engine

This file contains the `GitHubRepoAnalyzer` class. It manages loading code, splitting documents, creating the vector store, and answering questions.

#### `__init__(repo_path)`

This constructor stores the repository path and initializes the main components:

- `ChatGroq` is used to call the LLM. The current code sets `model="openai/gpt-oss-120b"` and `temperature=0.2`.
- `GoogleGenerativeAIEmbeddings` uses `gemini-embedding-001` to create vectors.
- `vector_store` starts as `None`.
- `rag_chain` starts as `None`.

The vector store and chain are initialized later, after documents have been processed.

#### `load_and_chunk_code()`

This method reads supported files and splits them into chunks.

**Step 1: Choose supported file types**

The `extensions` list contains glob patterns for files such as Python, JavaScript, TypeScript, Java, C/C++, Go, Rust, HTML, CSS, Markdown, SQL, YAML, TOML, shell scripts, Dockerfiles, and other configuration files.

For example:

```python
extensions = [
    "**/*.py",
    "**/*.js",
    "**/*.jsx",
    "**/*.ts",
    "**/*.java",
    "**/*.cpp",
    "**/*.md",
    "**/*.sql",
]
```

The actual list in the project is longer. A pattern such as `**/*.py` means Python files in matching folders, not only files in the repository's root.

**Step 2: Load matching files**

For each pattern, the code creates a `DirectoryLoader` with `TextLoader` as its loader class.

- `DirectoryLoader` searches the repository for matching paths.
- `TextLoader` reads each matching file as text.
- `loader.load()` returns LangChain `Document` objects.

A `Document` normally contains `page_content` and `metadata`. Here, `page_content` holds the file text and `metadata` can hold information such as the source path.

**Step 3: Filter unwanted content**

The code checks for directory names such as `.git`, `node_modules`, `.venv`, `venv`, `dist`, `build`, `.next`, and `coverage`. It also skips files whose basename starts with `test_`.

Note: this filtering happens after the loader returns documents. It is not a complete security or file-exclusion boundary, and the current test-file rule does not catch every possible test naming convention.

**Step 4: Add file information**

The method adds the filename to metadata and prefixes the file's content with its name and path:

```python
doc.page_content = (
    f"FILE: {filename}\n"
    f"PATH: {source}\n\n"
    f"{doc.page_content}"
)
```

This helps the LLM understand where a retrieved code fragment came from. The filename and path are included in the text, while the metadata is also carried with the document.

**Step 5: Split files into chunks**

The method uses `RecursiveCharacterTextSplitter` with:

```python
chunk_size=1500
chunk_overlap=200
```

These values are **characters**, not tokens.

The splitter tries separators in a priority order, including class/function-like boundaries, blank lines, newlines, spaces, and finally individual characters. If a file is too long, it is divided into smaller pieces.

- `chunk_size=1500`: aims for chunks of about 1,500 characters.
- `chunk_overlap=200`: allows neighboring chunks to share about 200 characters, preserving some context at split boundaries.

The splitter is not a full programming-language parser. It attempts to split at useful text boundaries but cannot guarantee that every chunk contains a complete function or class.

If no supported documents are found, the method raises a `ValueError`.

#### `build_vector_database(chunked_docs)`

This method creates the vector store.

The project uses:

```python
BATCH_SIZE = 20
WAIT_TIME = 20
```

The chunks are processed in batches of up to 20. The first batch creates a Chroma collection with a unique name. Later batches are added to the same vector store. Between batches, the code waits 20 seconds when more chunks remain.

The delay is a simple way to reduce the chance of hitting embedding API rate limits. It is a fixed delay, not an adaptive throttling algorithm.

Once all batches have been processed, `_initialize_chain()` is called.

#### `_initialize_chain()`

This method connects the retriever, prompt, and LLM.

The retriever is configured as follows:

```python
retriever = self.vector_store.as_retriever(
    search_kwargs={"k": 10}
)
```

The value `k=10` tells the retriever to return up to ten relevant chunks for a query.

The system prompt tells the LLM to explain the repository, use the retrieved context, avoid inventing functionality, and mention when information is missing. It also asks about areas such as architecture, files, classes, functions, authentication, database use, API routes, and execution flow.

The code then creates a document-combination chain and connects it to the retriever with `create_retrieval_chain()`.

#### `ask_question(question)`

This method checks that the RAG chain has been initialized. If it has not, the method raises an error. Otherwise, it invokes the chain with the user's question and returns the generated answer.

```python
response = self.rag_chain.invoke({"input": question})
return response["answer"]
```

### 3. `app.py` — Streamlit user interface

This file connects the user interface to the RAG engine.

#### Page setup

Streamlit sets the page title, icon, layout, and introduction. The sidebar contains a GitHub URL input plus **Analyze** and **Clear** buttons.

#### Session state

The app uses `st.session_state` to keep important values across Streamlit reruns:

- `analyzer`: the initialized analyzer object.
- `processed`: whether analysis completed successfully.
- `chat_history`: earlier user questions and assistant answers.
- `chunk_count`: number of chunks produced.
- `cloned_repo_path`: location of the temporary clone.

Streamlit reruns the script after many interactions. Session state helps retain these values between reruns.

#### Analyze button

When the user clicks **Analyze**, the app:

1. Checks that `GROQ_API_KEY` exists.
2. Checks that the URL is not empty.
3. Checks that the URL starts with `https://github.com/`.
4. Clones the repository.
5. Creates a `GitHubRepoAnalyzer`.
6. Loads and chunks the files.
7. Builds the vector database and RAG chain.
8. Saves the analyzer and chunk count in session state.
9. Displays a success message.

If an exception occurs during processing, the app attempts to delete the temporary clone and displays the exception.

The code checks for the Groq key explicitly in the UI. The Google API key is also needed by the embedding integration and can cause an error during indexing if it is missing.

#### Chat interface

After processing, the app displays earlier messages and provides `st.chat_input()` for a new question. It calls `ask_question()`, displays the answer, and adds it to chat history.

#### Clear button

The Clear button attempts to delete the temporary repository, resets the session state, and calls `st.rerun()` to refresh the interface.

---

## 1.5 Engineering Challenges and Solutions

These are useful engineering topics to discuss in an interview. Describe them as implementation decisions rather than claiming measured improvements that have not been tested.

### Challenge 1: Embedding API rate limits

**Problem:** Embedding services may limit how many requests or how much text can be processed during a time period. Sending too much content too quickly can cause rate-limit errors.

**Current approach:** The code processes up to 20 chunks per batch and waits 20 seconds between batches.

```python
BATCH_SIZE = 20
WAIT_TIME = 20
```

**Possible production improvement:** Use retries with exponential backoff and jitter, configurable rate limits, and a background task queue. The correct batch size and delay depend on the API quota and request sizes.

**Interview answer:**

> “I process chunks in batches and add a delay between batches to reduce rate-limit errors from the embedding API. For production, I would add retry handling with exponential backoff and move indexing into background workers.”

### Challenge 2: Splitting code without losing too much context

**Problem:** If a file is split at arbitrary points, a function or logical block can be separated across chunks.

**Current approach:** The splitter tries code-related separators such as `class`, `def`, `function`, and `export` before falling back to newlines and spaces.

**Limitation:** This is still text-based splitting, not true syntax-aware parsing. It does not guarantee that every chunk is a complete function.

**Possible improvement:** Use a parser such as Tree-sitter to identify function and class boundaries.

### Challenge 3: Keeping file names and paths visible

**Problem:** A retrieved code fragment may be useful but difficult to locate in the original repository.

**Current approach:** The code prefixes each document with `FILE:` and `PATH:` and also stores the filename in metadata.

**Why it helps:** The file location becomes part of the text sent to the embedding model and, if retrieved, the LLM. It can help the answer mention the relevant file.

**Limitation:** This improves location awareness but does not guarantee perfectly accurate citations or line numbers.

### Challenge 4: Avoiding irrelevant files

**Problem:** Generated files, virtual environments, and other irrelevant content can make retrieval noisier and increase indexing work.

**Current approach:** The code has an allowlist of glob patterns and filters paths containing certain directory names. It skips files whose names start with `test_`.

**Limitation:** Some unwanted files can still pass through. For example, `example.test.js` does not start with `test_`, and the current ignored-directory list does not include every possible test directory.

**Possible improvement:** Exclude more test naming conventions, apply directory exclusions during loading, limit repository size, and filter generated or minified files.

### Challenge 5: Cleaning up temporary repositories

**Problem:** Cloned repositories take disk space. If they are not removed, repeated analyses can leave unnecessary files behind.

**Current approach:** The clone path is stored in session state. The app calls `delete_repository()` when the user clicks Clear and when certain processing errors occur.

**Limitation:** A server crash or abrupt termination may prevent normal cleanup. The current cleanup also focuses on the cloned directory, not on adding persistent storage or lifecycle management for a production deployment.

**Possible improvement:** Add periodic cleanup of old temporary folders, per-user quotas, and background-job lifecycle management.

---

## 1.6 Explaining Metrics Honestly

Interviewers may ask how you measured performance. Only use numbers you actually measured and can explain.

| Claim | What you should be able to show |
|---|---|
| Repository size supported | The repository URL, file count, or line count used in a test |
| Number of chunks indexed | The value of `len(chunks)` or a saved test log |
| Query response time | Timings measured over several queries, including network/API time |
| Reduction in onboarding time | A repeatable comparison of manual exploration against use of the application |
| Number of users supported | A concurrency test or deployment evidence |

Do not present estimated figures as measured results. For example, a response time below two seconds cannot be guaranteed merely because ChromaDB is in memory or because Groq is fast. API latency, prompt size, and answer length also affect response time.

A safe interview response is:

> “The current implementation demonstrates the end-to-end RAG workflow. I have not treated unmeasured performance estimates as benchmark results. I would measure indexing duration, query latency, retrieval quality, and answer correctness with a repeatable test set before making quantitative performance claims.”

---

# PART 2: Technology Choices and Code Details

## 2.1 Why These Technologies?

### 1. Why ChromaDB?

**Why it fits this project:**

- It can store embeddings and associated document information.
- It supports similarity search.
- It integrates with LangChain.
- It is convenient for a small prototype without operating a separate vector database service.

**Why not FAISS?**

FAISS is a library for efficient vector similarity search. ChromaDB offers a more database-like interface for collections and document metadata, which is convenient for this project.

**Why not Pinecone?**

Pinecone is a managed cloud vector database. It can be useful for production systems, but a hosted service may introduce cost, credentials, and network dependencies that are unnecessary for a small local prototype.

**When might another database be useful?**

For a larger production application, a persistent or distributed vector database such as Qdrant or Milvus may be worth considering. The best choice depends on scale, persistence, filtering, operations, and cost.

### 2. Why Groq for the LLM?

The project uses LangChain's `ChatGroq` integration to call a model hosted through Groq. The model name in the current code is `openai/gpt-oss-120b`.

**Why it fits:**

- The app can call a hosted model through an API.
- It does not need to run the large model on the user's computer.
- LangChain provides a convenient interface for using the model in the chain.

**Why not a local model?**

A local model avoids depending on a hosted inference API, but larger models can require substantial memory and compute resources. A smaller local model may be an option when privacy, offline use, or predictable local operation is important.

**Note:** Model availability, speed, and cost can change. Check the provider's current documentation before making exact speed or price comparisons.

### 3. Why Gemini Embeddings?

The project uses `GoogleGenerativeAIEmbeddings` with the model name `gemini-embedding-001`.

Embeddings convert text into numerical vectors. The system uses them so that code chunks and user questions can be compared by semantic similarity.

**Why separate embeddings and an LLM?**

They perform different jobs:

- The **embedding model** creates vectors for searching.
- The **LLM** uses the question and retrieved context to generate a natural-language answer.

The code uses Gemini for embeddings and Groq for answer generation. This is a common design because the two components do not have to come from the same provider.

### 4. Why LangChain?

LangChain provides components that connect the different parts of the RAG workflow.

In this project it is used for:

- Loading documents.
- Splitting documents into chunks.
- Connecting to Gemini Embeddings.
- Connecting to ChromaDB.
- Creating a retriever.
- Building a prompt and retrieval chain.
- Calling the LLM through a common interface.

Without a framework, you could implement these steps using separate SDKs and custom code. LangChain reduces the amount of integration code, although it also introduces abstractions that developers need to understand.

### 5. Why Streamlit?

Streamlit makes it possible to build a simple Python web interface without separately writing a React frontend.

The project uses it for:

- The GitHub URL input.
- Analyze and Clear buttons.
- Progress messages and errors.
- Chat input and message display.
- Session state.

For a production application requiring a highly customized interface, a React or Next.js frontend with a separate API backend could be a better fit.

### 6. Why GitPython?

GitPython lets Python code interact with Git repositories. The project uses `Repo.clone_from()` to clone a repository into a temporary directory.

The helper function also uses Python's `tempfile` and `shutil` modules:

- `tempfile` creates a temporary directory.
- GitPython clones the repository into it.
- `shutil.rmtree()` removes the directory during cleanup.

### 7. Why `DirectoryLoader` and `TextLoader`?

`DirectoryLoader` searches for files that match the specified glob pattern. `TextLoader` reads each matched file as text.

Using multiple patterns allows the project to load supported source code and configuration files without trying to process every file in the repository.

### 8. Why an in-memory vector store?

The current code creates a Chroma collection with a unique name but does not configure a persistent storage directory. The vector store is therefore not set up as a durable, reusable index.

This is convenient for a prototype, but it means a new application run may need to index the repository again. A production design would consider persistent storage, cache invalidation, repository versions, and cleanup policies.

---

## 2.2 Important Functions and Parameters

### 1. `RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)`

#### `chunk_size=1500`

This controls the approximate size of a chunk in **characters** for this splitter. It does not mean 1,500 tokens.

Character count and token count are different. The number of tokens depends on the language, symbols, and tokenizer.

#### `chunk_overlap=200`

Adjacent chunks may share about 200 characters. This helps preserve some context when a useful code block crosses a chunk boundary.

#### Why use a splitter at all?

A whole repository can be too large to embed or provide to the LLM as one document. Chunks make it possible to index and retrieve smaller pieces.

#### Limitation

A fixed character size does not guarantee that a chunk contains a complete function. The separator list improves the chance of splitting at useful boundaries, but it is not an AST parser.

### 2. `as_retriever(search_kwargs={"k": 10})`

The code creates a retriever from the Chroma vector store.

`k=10` means the retriever asks for up to ten relevant document chunks for a query.

Why ten? It is a design choice intended to provide context from more than one file. A smaller value may miss useful evidence; a larger value may add irrelevant context and increase prompt size. The best value should be evaluated using real questions.

### 3. `create_stuff_documents_chain()`

This chain combines retrieved documents with a prompt and sends them to the LLM.

The term **stuff** means that the retrieved document content is placed into the prompt context together.

A simplified view:

```text
Retrieved chunks
       +
User question
       +
System instructions
       |
       v
Prompt sent to the LLM
       |
       v
Generated answer
```

The more context that is included, the more important it becomes to manage prompt size and relevance.

### 4. `create_retrieval_chain()`

This connects the retriever to the document-combination chain.

In simple terms, it coordinates the process:

1. Receive the user's question.
2. Retrieve relevant documents.
3. Pass those documents to the answer chain.
4. Return the resulting answer and other chain outputs.

### 5. `temperature=0.2`

Temperature influences how varied the LLM's token choices can be.

- Lower values generally produce more focused, less varied answers.
- Higher values generally allow more varied answers.

A low temperature can be useful for code explanations, but it does **not** guarantee factual correctness or eliminate hallucinations.

### 6. Similarity search: a beginner explanation

An embedding is a list of numbers representing text. The embedding model is trained so that useful semantic relationships can be represented in vector space.

When a question is asked:

1. The question is embedded into a vector.
2. ChromaDB compares that vector with stored vectors.
3. The retriever returns chunks that are close according to the configured distance metric.

The exact distance metric depends on the vector-store configuration. Do not claim that this project explicitly uses cosine distance unless you have verified the Chroma collection's configuration.

### 7. Metadata

Metadata is information attached to a document, such as its source path or filename.

The project stores a `filename` metadata field and retains the loader's source metadata. The code also places the filename and path directly into `page_content`.

This can help the LLM identify the file associated with a retrieved chunk. Metadata can also be useful for filtering or displaying source information in more advanced versions.

---

# PART 3: 50 Interview Questions and Answers

The answers below are deliberately short enough to practise aloud. Understand the idea rather than memorizing every sentence.

## Category A: Project Overview (Q1–Q8)

### Q1. Can you explain your project?

**Answer:**

> “It is a RAG-based GitHub Repository Analyzer. A user provides a public repository URL, the app indexes supported code and text files into ChromaDB using Gemini embeddings, and a retriever finds relevant chunks when a question is asked. A Groq-hosted LLM then generates an answer using those chunks as context.”

### Q2. What problem does it solve?

**Answer:**

> “It helps developers explore unfamiliar codebases more easily. Instead of manually searching many files, they can ask natural-language questions about the architecture, functions, routes, and execution flow.”

### Q3. What is RAG?

**Answer:**

> “RAG stands for Retrieval-Augmented Generation. It retrieves relevant information from an external knowledge base and provides that information to an LLM as context before the model generates an answer.”

### Q4. Why did you use RAG instead of an LLM alone?

**Answer:**

> “A standalone LLM may not know the private or latest details of a particular repository. RAG lets us retrieve the repository's current indexed content without retraining the model whenever the knowledge base changes.”

### Q5. What are the main limitations of your current implementation?

**Answer:**

> “The vector index is not configured for durable persistence, the splitter is text-based rather than syntax-aware, and retrieval can miss relevant code when a question requires connections across multiple files. The embedding batch delay can also make initial indexing slow.”

### Q6. What would you improve if you rebuilt it?

**Answer:**

> “I would add syntax-aware chunking with Tree-sitter, stronger hybrid retrieval, source citations, persistent indexing, and automated evaluation. I would also move long indexing jobs to a background queue.”

### Q7. Does it support private repositories?

**Answer:**

> “The current clone helper does not implement GitHub authentication, so it is designed for public repositories. Private repository support would require secure OAuth or token-based authentication, careful secret handling, and authorization checks.”

### Q8. What does GitPython do?

**Answer:**

> “GitPython lets Python code interact with Git. I use it to clone a public GitHub repository into a temporary local directory so the application can read and analyze its files.”

---

## Category B: Architecture and Design (Q9–Q18)

### Q9. Walk through the life of a user question.

**Answer:**

> “Streamlit captures the question and calls `ask_question()`. The retrieval chain searches ChromaDB for relevant chunks, puts the chunks into the prompt with the question, calls the LLM through Groq, and returns the answer to Streamlit.”

### Q10. Why use ChromaDB instead of a normal relational database?

**Answer:**

> “ChromaDB provides a convenient interface for storing embeddings and performing vector similarity search. A relational database can also support vector search with an extension such as pgvector, but ChromaDB is straightforward for this prototype.”

### Q11. What is the difference between the knowledge base and the LLM?

**Answer:**

> “The knowledge base contains the indexed repository content and its embeddings. The LLM is the model that uses the question and retrieved content to generate a natural-language answer.”

### Q12. What is the difference between indexing and retrieval?

**Answer:**

> “Indexing prepares the repository by loading files, splitting them into chunks, generating embeddings, and storing them. Retrieval happens when a user asks a question and the system searches for relevant chunks.”

### Q13. What happens if no supported source files are found?

**Answer:**

> “The loader method raises a `ValueError` if no documents are found. The UI catches processing errors and displays the exception, while attempting to clean up the temporary repository.”

### Q14. Where is the repository stored temporarily?

**Answer:**

> “`tempfile.mkdtemp()` creates a temporary directory using the operating system's configured temporary location. GitPython clones the repository there. The Clear button and certain error paths call `shutil.rmtree()` to remove it.”

### Q15. Why are `chunk_size=1500` and `chunk_overlap=200` used?

**Answer:**

> “The splitter aims for chunks of about 1,500 characters, with around 200 characters shared between adjacent chunks. This gives the retriever smaller pieces to search and helps retain context around split boundaries. These are characters, not tokens.”

### Q16. Why add `FILE:` and `PATH:` to the document?

**Answer:**

> “It gives the retrieved text information about its original file and path. This can help the LLM explain where a code fragment came from instead of discussing it without location context.”

### Q17. What does `temperature=0.2` mean?

**Answer:**

> “It is a low temperature setting that generally makes output less varied. It can be useful for focused code explanations, but it does not guarantee correctness.”

### Q18. What is the ‘Lost in the Middle’ problem?

**Answer:**

> “It describes a tendency for some LLMs to use information less reliably when it is buried in the middle of a very long context. RAG reduces the amount of context by retrieving selected chunks instead of sending the whole repository each time.”

---

## Category C: Code and LangChain (Q19–Q28)

### Q19. What does `RecursiveCharacterTextSplitter` do?

**Answer:**

> “It splits long text into smaller chunks. It tries separators in a priority order, starting with more meaningful boundaries and falling back to newlines, spaces, and characters when necessary.”

### Q20. What is the difference between `create_stuff_documents_chain` and `create_retrieval_chain`?

**Answer:**

> “The stuff chain combines documents with the prompt and calls the LLM. The retrieval chain connects a retriever to that answer chain, so it can fetch relevant documents for the user's question first.”

### Q21. Why use `st.session_state`?

**Answer:**

> “Streamlit reruns the script after many interactions. Session state stores values such as the analyzer, chat history, processing status, and clone path across those reruns.”

### Q22. What does `shutil.rmtree()` do?

**Answer:**

> “It recursively deletes a directory and its contents. The project uses it to remove the temporary repository folder during cleanup.”

### Q23. Why use a unique Chroma collection name?

**Answer:**

> “The code generates a unique collection name with a UUID for each new vector store. This helps keep chunks from different repository analyses separate instead of accidentally mixing them in one collection.”

### Q24. What are `DirectoryLoader` and `TextLoader`?

**Answer:**

> “`DirectoryLoader` finds files matching a glob pattern. `TextLoader` reads each matching file as text and returns it as a LangChain document.”

### Q25. Why use `load_dotenv()`?

**Answer:**

> “It loads key-value pairs from a `.env` file into environment variables. This lets the application access API keys without hardcoding them in the Python source files.”

### Q26. Why wait 20 seconds after a batch?

**Answer:**

> “The code pauses between batches to reduce the chance of exceeding embedding API rate limits. It is a fixed delay, not a dynamic rate-limit controller.”

### Q27. What does `as_retriever(search_kwargs={"k": 10})` do?

**Answer:**

> “It exposes the Chroma vector store through LangChain's retriever interface. The `k=10` setting asks it to return up to ten relevant chunks for a question.”

### Q28. How does the prompt try to reduce hallucinations?

**Answer:**

> “It instructs the LLM to use the retrieved repository context, avoid inventing functionality, and state when the retrieved context does not contain the answer. These instructions help but cannot completely prevent hallucinations.”

---

## Category D: RAG and Vector Databases (Q29–Q38)

### Q29. What is an embedding?

**Answer:**

> “An embedding is a numerical vector representing text. Text with related meaning may have vectors that are close to each other, which helps the system find semantically relevant chunks.”

### Q30. Which distance metric does your vector store use?

**Answer:**

> “The code uses ChromaDB through LangChain but does not explicitly set the distance metric in the shown configuration. Chroma supports different metrics, so I would verify the collection configuration before naming the exact metric.”

### Q31. What is HNSW?

**Answer:**

> “HNSW stands for Hierarchical Navigable Small World. It is a graph-based method used by some vector indexes to find approximate nearest neighbors efficiently. The exact index and settings depend on the vector database configuration.”

### Q32. What is the difference between RAG and a long-context LLM?

**Answer:**

> “A long-context model can accept a large amount of text, while RAG searches a knowledge base and sends selected information to the model. RAG can reduce irrelevant context and repeated input, although its quality depends on retrieving the right information.”

### Q33. What is chunk overlap?

**Answer:**

> “Chunk overlap is the amount of text shared between neighboring chunks. It helps preserve context when a useful piece of logic is split near a chunk boundary.”

### Q34. What are context recall and context precision?

**Answer:**

> “Context recall asks whether the retriever found the relevant information needed for an answer. Context precision asks how much of the retrieved information is actually relevant rather than noise.”

### Q35. How does the app answer a broad question about architecture?

**Answer:**

> “The loader includes documentation and configuration files as well as source code. Semantic retrieval may find useful project descriptions and code, but a broad question can still require information from more files than the retriever returns.”

### Q36. What is dense retrieval versus sparse retrieval?

**Answer:**

> “Dense retrieval uses embeddings to find semantically related text. Sparse retrieval uses keyword-based methods such as BM25 and is useful for exact names or identifiers. Hybrid search combines both.”

### Q37. What is re-ranking?

**Answer:**

> “Re-ranking takes an initial list of retrieved candidates and scores them again with a model or another relevance method. It can improve the order of results before the final chunks are sent to the LLM.”

### Q38. What is prompt injection in a code-analysis RAG system?

**Answer:**

> “A repository may contain text that tries to manipulate the LLM, such as comments telling it to ignore its instructions. Retrieved code should be treated as untrusted data. A system prompt helps, but production systems also need clear context separation, security testing, and controls over tools and secrets.”

---

## Category E: Errors and Security (Q39–Q44)

### Q39. What happens if someone submits a malicious Git URL?

**Answer:**

> “The UI checks that the URL starts with `https://github.com/`, but that is only a basic validation. A production system should validate the host and URL structure more strictly, enforce resource limits, and isolate clone and parsing work. It should not rely on a prefix check as its only security control.”

### Q40. What if someone submits a huge repository?

**Answer:**

> “Cloning and indexing a very large repository could consume too much disk, memory, time, or API quota. Production safeguards could include repository size limits, timeouts, shallow clones, file-count limits, and background workers.”

### Q41. How are binary files handled?

**Answer:**

> “The loader uses an allowlist of file patterns for source code and text-like files, rather than deliberately loading every file type. However, the allowlist alone does not guarantee that every matched file is valid UTF-8 text; loader errors and file size should also be handled carefully.”

### Q42. What happens if the user asks an unrelated question?

**Answer:**

> “The system prompt tells the LLM to answer from retrieved repository context and to say when information is unavailable. That may discourage unrelated answers, but the application should still test this behavior because the prompt cannot guarantee refusal in every case.”

### Q43. How does the project handle file encoding errors?

**Answer:**

> “The loader is configured to read files using UTF-8 and `silent_errors=True`. Some loading errors may therefore be skipped instead of stopping the whole process. This improves resilience but can also hide missing files unless errors are logged.”

### Q44. What is the risk of storing API keys in `.env`?

**Answer:**

> “If `.env` is committed or exposed, someone else may use the API keys. It should be excluded from version control, and exposed keys should be revoked. In production, a secret manager or deployment environment variables are usually preferable.”

---

## Category F: Scaling and Production (Q45–Q50)

### Q45. How would you scale the app for many concurrent users?

**Answer:**

> “I would separate the UI and API, move repository cloning and embedding into background workers, add a job queue, use persistent shared vector storage, and cache indexes for unchanged repositories. I would also add per-user limits, monitoring, and retries.”

### Q46. How would you update the index when a new commit is pushed?

**Answer:**

> “A GitHub webhook could notify the application about a push. The app could identify changed files, remove or replace their old chunks, and embed only the updated content. It would also need to handle deleted files and failed updates.”

### Q47. How would you implement AST-based chunking?

**Answer:**

> “I would use a parser such as Tree-sitter to identify functions, classes, and other syntax nodes. Each chunk could then correspond to a meaningful code unit rather than relying only on character boundaries.”

### Q48. How would you add hybrid search?

**Answer:**

> “I would combine dense vector retrieval for semantic questions with sparse keyword retrieval for exact function names and identifiers. A rank-fusion method could combine the two result lists before sending the best chunks to the LLM.”

### Q49. What monitoring would you add?

**Answer:**

> “I would record indexing duration, query latency, embedding and LLM errors, token usage, API costs, and retrieval quality. Tools such as OpenTelemetry or a tracing platform could help identify slow or unsuccessful stages.”

### Q50. How would you test answer quality automatically?

**Answer:**

> “I would create a set of repositories and questions with verified expected answers or evidence. Then I would evaluate whether the right chunks were retrieved, whether answers were supported by those chunks, and whether they actually answered the question. These tests could run whenever the code or prompt changes.”

---

# Quick Interview Cheat Sheet

| Term or setting | Used in this project | Simple explanation |
|---|---|---|
| RAG | Retrieval-Augmented Generation | Retrieve relevant information before asking the LLM to answer |
| Repository clone | GitPython | Makes a local copy of a public GitHub repository |
| Temporary folder | `tempfile.mkdtemp()` | Stores the cloned repository temporarily |
| Cleanup | `shutil.rmtree()` | Deletes the temporary directory |
| File loading | `DirectoryLoader` + `TextLoader` | Finds matching files and reads them as text |
| Chunk size | `1500` characters | Approximate maximum text size used by the splitter |
| Chunk overlap | `200` characters | Shared text between neighboring chunks |
| Embedding model | `gemini-embedding-001` | Converts text into numerical vectors |
| Vector store | ChromaDB | Stores chunks and vectors for similarity search |
| Retriever | `k=10` | Requests up to ten relevant chunks |
| LLM integration | `ChatGroq` | Calls the configured model through Groq |
| Current model name | `openai/gpt-oss-120b` | Model configured in the current code |
| Temperature | `0.2` | Low-randomness generation setting |
| Batch size | `20` chunks | Maximum chunks passed in each indexing batch |
| Batch delay | `20` seconds | Fixed pause between batches when more remain |
| UI | Streamlit | Provides the URL input and chat interface |
| Session state | `st.session_state` | Keeps app data across Streamlit reruns |
| Indexing | Before asking questions | Reads, chunks, embeds, and stores repository content |
| Inference | When a question is asked | Retrieves context and generates the answer |
| Main limitation | Text-based retrieval | May miss relationships across files or code structure |
| Important caution | No guaranteed latency | Measure performance before claiming exact response times |

---

## Final 60-second recap

If you remember only one flow, remember this:

```text
Public GitHub URL
      ↓
GitPython clones the repository
      ↓
Load supported text/code files
      ↓
Split files into chunks
      ↓
Gemini creates embeddings
      ↓
ChromaDB stores the chunks and vectors
      ↓
User asks a question
      ↓
Retriever fetches relevant chunks
      ↓
Groq-hosted LLM receives question + context
      ↓
Streamlit displays the answer
```

**The most important interview distinction:** this project does not train or fine-tune an LLM on every repository. It indexes repository content and retrieves relevant pieces at question time. The model then uses those pieces as context to generate an answer.
