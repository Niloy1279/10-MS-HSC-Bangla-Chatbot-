# 10-MS-HSC-Bangla-Chatbot-
A RAG Pipeline project , where scan a bangla written pdf and based on that answer different types of questions. 
This project implements a Retrieval-Augmented Generation (RAG) system designed to answer questions about the HSC Bangla 1st Paper curriculum. It leverages Google Gemini for generation, a Sentence Transformer for embeddings, and ChromaDB for vector storage, allowing users to query a PDF document in both English and Bengali.
Prerequisites:
    Python 3.9+
    A Google API Key for the Gemini API. You can obtain one from Google AI Studio.

Install the required Python packages:(pip install -q -U langchain langchain-google-genai langchain-community chromadb pymupdf sentence-transformers google-ai-generativelanguage==0.6.15
)    


Used Tools, Libraries, and Packages

langchain: The primary framework used for building applications with large language models, providing tools for chaining components.

langchain-google-genai: Enables integration with Google's Generative AI models, specifically used here for ChatGoogleGenerativeAI (Gemini 1.5 Flash).

langchain-community: Provides various community-contributed LangChain components, including SentenceTransformerEmbeddings and Chroma for vector store interactions.

chromadb: An open-source vector database used for storing and efficiently querying the embedded document chunks.

pymupdf (fitz): A robust Python library for PDF manipulation, used for extracting text from the PDF document.

sentence-transformers: A library for state-of-the-art sentence, paragraph, and image embeddings, specifically used for the paraphrase-multilingual-MiniLM-L12-v2 model.

google-ai-generativelanguage: Google's official client library for interacting with their Generative AI models, underlying langchain-google-genai.

Standard Python Libraries (os, re, getpass, asyncio, typing): Used for file system interactions, regular expressions for text cleaning, secure password input, asynchronous operations, and type hinting respectively.


API Documentation
This project is implemented as a command-line interface (CLI) application and does not expose a separate REST API for external consumption. All interactions occur directly through the Python script.
The core API interactions with Google's Generative AI models are handled internally by the langchain-google-genai library. This library abstracts away the direct fetch calls to the Google Gemini API, managing authentication and request/response formatting behind the scenes. Users only need to provide their GOOGLE_API_KEY as an environment variable or via the getpass prompt within the script.



Concise Answers to RAG System Questions
PDF Text Extraction:

Method/Library: Used PyMuPDF (imported as fitz) for text extraction (page.get_text("text")).

Why: Chosen for its efficiency and ability to extract raw text, which was then processed.

Challenges: Faced issues with hyphenated words across lines, irregular whitespace, and non-standard characters (especially in Bengali text). These were addressed using regular expressions in the clean_text function to normalize the text and preserve word integrity.

Chunking Strategy:

Strategy: RecursiveCharacterTextSplitter with a character limit (chunk_size=700) and overlap (chunk_overlap=100).

Why it works well: This approach prioritizes splitting on natural breaks (paragraphs, lines) while ensuring context continuity with overlap. This helps maintain semantically coherent units, which is crucial for accurate retrieval, especially with varying sentence structures in multilingual text.

Embedding Model:

Model: paraphrase-multilingual-MiniLM-L12-v2.

Why chosen: Selected for its strong multilingual capabilities (supporting Bengali and English), its fine-tuning for paraphrase and semantic similarity tasks, and its efficiency (being a "MiniLM" model).

How it captures meaning: It transforms text into numerical vectors where sentences with similar meanings (regardless of language) are mapped to close points in the embedding space, enabling cross-lingual semantic comparison.

Query Comparison and Storage:

Comparison: Queries are embedded into vectors using the same model as the document chunks. ChromaDB then performs a cosine similarity search to find document chunks whose vectors are most geometrically similar to the query vector.

Why chosen: Cosine similarity is effective for measuring directional semantic similarity between embeddings. ChromaDB was chosen for its ease of integration, efficient vector storage, and persistence, which avoids re-embedding the entire document every time.

Meaningful Comparison and Vague Queries:

Meaningful Comparison: Ensured by using the same embedding model for both document chunks and queries, placing them in the same semantic vector space.

Vague/Missing Context Queries: If a query is vague, the retrieval might be broad. However, the chat_history in the RAG chain's prompt helps the LLM contextualize the current question based on previous turns, making the query more specific for retrieval and improving the LLM's ability to provide a relevant answer. If context is truly insufficient, the LLM is explicitly instructed by the prompt to state that it doesn't know the answer, preventing fabrication.

Relevance and Improvements:

Relevance: The results generally seem relevant for questions directly answerable by the document, thanks to the multilingual embedding and robust cleaning/chunking.

Improvements:

Better Chunking: Explore semantic chunking or adaptive chunk sizing to ensure more cohesive context.

Better Embedding Model: Consider larger or domain-specific multilingual embedding models if higher accuracy is needed and resources allow.

Advanced Retrieval: Implement re-ranking of retrieved documents or hybrid search (combining vector and keyword search) to improve precision and recall.

Source Attribution: (Advanced) Indicate which specific document chunks or pages contributed to the answer for user verification.
