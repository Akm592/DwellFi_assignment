# Memory-Persistent AI Research Assistant

This project is a memory-persistent AI research assistant designed to answer questions about a specific document (`adobe-annual-report.pdf`) while maintaining conversational context. The assistant is built using the LlamaIndex framework and is exposed as a FastAPI application.

## Core Features

*   **Memory System:**
    *   **Short-Term Memory:** Remembers the immediate conversation history within a session.
    *   **Long-Term Memory:** Extracts and stores key facts and research topics across sessions.
*   **Query Processing:**
    *   **Query Planning:** Complex questions are broken down into smaller, manageable sub-queries.
    *   **Retrieval-Augmented Generation (RAG):** The assistant uses a RAG pipeline to find relevant information in the provided PDF and generate answers.
*   **Tools:**
    *   **Keyword Extraction:** Identifies key terms in a text using YAKE and KeyBERT.
    *   **Summarization:** Creates summaries of documents or conversations.
*   **Technology Stack:**
    *   **Backend:** FastAPI (Python)
    *   **AI/LLM Framework:** LlamaIndex
    *   **Vector Database:** ChromaDB
    *   **LLM:** Groq
    *   **Containerization:** Docker and Docker Compose

## Project Structure

The project is organized into the following main directories:

*   `src/`: Contains the core application logic.
    *   `memory/`: Manages short-term and long-term memory.
    *   `retrieval/`: Handles loading documents and querying the vector store.
    *   `tools/`: Contains additional functionalities like keyword extraction and summarization.
    *   `utils/`: Includes configuration, logging, and other utility functions.
    *   `workflows/`: Defines the main logic for processing queries and planning complex questions.
    *   `app.py`: The main FastAPI application file.
*   `data/`: Stores the research document (`adobe-annual-report.pdf`).
*   `tests/`: Contains unit and integration tests.
*   `chroma_db/`: The persistent storage for the ChromaDB vector store.

## How It Works: A Step-by-Step Flow

1.  **User sends a query:** A user sends a question to the `/query` endpoint of the FastAPI application.
2.  **Query Complexity Check:** The `MainResearchWorkflow` assesses if the query is simple or complex.
3.  **Query Processing:**
    *   **Simple Queries:** The query is directly sent to the query engine, which retrieves relevant information from the PDF and generates an answer.
    *   **Complex Queries:** The `QueryPlanningWorkflow` breaks the query into sub-queries. Each sub-query is executed, and the results are combined to form a comprehensive answer.
4.  **Memory Update:**
    *   The conversation (user query and assistant's response) is stored in the **short-term memory**.
    *   Key information and research topics are extracted and saved in the **long-term memory**.
5.  **Response:** The final answer is sent back to the user.

## Key Architectural Decisions

*   **LlamaIndex Workflows:** The use of LlamaIndex's workflow framework provides a structured and modular way to organize the different stages of the research process.
*   **FastAPI:** FastAPI is a good choice for the web framework due to its high performance and ease of use.
*   **ChromaDB:** ChromaDB is used for its simplicity and persistence, making it easy to store and retrieve document embeddings.
*   **Docker:** Docker ensures that the application runs in a consistent and reproducible environment, which is great for both development and deployment.
*   **Modular Design:** The code is well-organized into modules, which makes it easier to understand, maintain, and extend.

## Setup and Deployment

### Prerequisites

*   Python 3.9+
*   Docker (optional, for containerized deployment)
*   Docker Compose (optional, for containerized deployment)

### Local Development Setup

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd memory-research-assistant
    ```

2.  **Create a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Create a `.env` file:**
    Create a `.env` file in the root of the project and add your `GROQ_API_KEY`:
    ```
    GROQ_API_KEY="your-groq-api-key"
    ```

5.  **Run the application:**
    ```bash
    uvicorn src.app:app --reload
    ```
    The application will be available at `http://127.0.0.1:8000`.

### Dockerized Deployment

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd memory-research-assistant
    ```
2.  **Create a `.env` file:**
    Create a `.env` file in the root of the project and add your `GROQ_API_KEY`:
    ```
    GROQ_API_KEY="your-groq-api-key"
    ```
3.  **Build and run the Docker container:**
    ```bash
    docker-compose up --build
    ```
    The application will be available at `http://127.0.0.1:8000`.

## Running Tests

To run the unit and integration tests, you can execute the following command in the root of the project:

```bash
pytest
```

## Evaluation

The `evaluate_agent.py` script can be used to evaluate the performance of the agent.

1.  **Ensure the agent is running:**
    Make sure the application is running in a separate terminal using `docker-compose up`.
2.  **Run the evaluation script:**
    ```bash
    python evaluate_agent.py
    ```
    The script will send a series of questions to the agent and score the responses for faithfulness and relevancy, printing a summary of the results at the end.