<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>RAG Chatbot with LLM</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            line-height: 1.6;
            margin: 40px;
            background-color: #f8f9fa;
            color: #212529;
        }
        h1, h2, h3 {
            color: #0d6efd;
        }
        code, pre {
            background-color: #e9ecef;
            padding: 8px;
            border-radius: 5px;
            display: block;
            overflow-x: auto;
        }
        ul {
            margin-left: 20px;
        }
        .section {
            margin-bottom: 30px;
        }
        .footer {
            margin-top: 40px;
            font-size: 0.9em;
            color: #555;
        }
    </style>
</head>
<body>

<h1>📚 RAG Chatbot with LLM </h1>

<div class="section">
    <p>
        A <strong>Retrieval-Augmented Generation (RAG) chatbot</strong> built using
        <strong>Streamlit</strong>, <strong>ChromaDB</strong>, and a <strong>local LLM (Llama.cpp)</strong>.
        This application allows users to upload Markdown documents, index them into a vector database,
        and ask questions grounded in the uploaded content.
    </p>
</div>

<div class="section">
    <h2>🚀 Features</h2>
    <ul>
        <li>Local LLM inference using Llama.cpp</li>
        <li>Vector similarity search with ChromaDB</li>
        <li>Upload and index Markdown documents</li>
        <li>Context-aware question answering (RAG)</li>
        <li>Multiple context synthesis strategies</li>
        <li>Chat history management</li>
        <li>Streaming token responses</li>
        <li>Configurable chunk size and overlap</li>
        <li>Persistent vector database storage</li>
    </ul>
</div>

<div class="section">
    <h2>🧠 Architecture Overview</h2>
    <pre>
User Query
   ↓
Question Refinement (LLM)
   ↓
Vector Search (ChromaDB)
   ↓
Context Synthesis Strategy
   ↓
LLM Answer Generation
   ↓
Streamlit UI
    </pre>
</div>

<div class="section">
    <h2>🗂️ Project Structure</h2>
    <pre>
rag_chatbot/
│
├── bot/
│   ├── client/               # LLM client (Llama.cpp)
│   ├── conversation/         # Chat history & context handling
│   ├── memory/               # Embedding + vector DB logic
│   ├── model/                # Model registry
│
├── document_loader/
│   ├── format.py
│   ├── text_splitter.py
│
├── entities/
│   └── document.py
│
├── helpers/
│   ├── log.py
│   └── prettier.py
│
├── models/                   # Local LLM models
├── vector_store/             # Persistent ChromaDB index
├── images/
│   └── bot.png
│
├── rag_chatbot_app.py
└── README.html
    </pre>
</div>

<div class="section">
    <h2>⚙️ Installation</h2>
    <h3>1. Clone Repository</h3>
    <pre>
git clone https://github.com/your-username/rag-chatbot.git
cd rag-chatbot
    </pre>

    <h3>2. Create Virtual Environment</h3>
    <pre>
python -m venv venv
source venv/bin/activate   (Linux/Mac)
venv\Scripts\activate      (Windows)
    </pre>

    <h3>3. Install Dependencies</h3>
    <pre>
pip install -r requirements.txt
    </pre>
</div>

<div class="section">
    <h2>🤖 Model Setup</h2>
    <ol>
        <li>Download a GGUF model compatible with Llama.cpp (e.g. Mistral, Llama2).</li>
        <li>Place the model file inside:</li>
    </ol>
    <pre>
models/
    </pre>
</div>

<div class="section">
    <h2>▶️ Running the Application</h2>
    <pre>
streamlit run rag_chatbot_app.py
    </pre>

    <h3>Optional Parameters</h3>
    <pre>
streamlit run rag_chatbot_app.py -- \
  --model mistral \
  --k 3 \
  --max-new-tokens 512 \
  --chunk-size 1000 \
  --chunk-overlap 50
    </pre>
</div>

<div class="section">
    <h2>📄 Document Upload</h2>
    <ul>
        <li>Supported format: Markdown (.md)</li>
        <li>Upload via sidebar</li>
        <li>Documents are chunked, embedded, and stored in ChromaDB</li>
        <li>Used for context-aware question answering</li>
    </ul>
</div>

<div class="section">
    <h2>🧪 Example Workflow</h2>
    <ol>
        <li>Upload Markdown documents</li>
        <li>Ask a question related to the documents</li>
        <li>Relevant chunks are retrieved</li>
        <li>LLM generates an answer using retrieved context</li>
        <li>Chat history is preserved</li>
    </ol>
</div>

<div class="section">
    <h2>🛠️ Configuration Parameters</h2>
    <table border="1" cellpadding="8">
        <tr>
            <th>Parameter</th>
            <th>Description</th>
            <th>Default</th>
        </tr>
        <tr>
            <td>--model</td>
            <td>LLM model name</td>
            <td>first available</td>
        </tr>
        <tr>
            <td>--k</td>
            <td>Number of retrieved chunks</td>
            <td>2</td>
        </tr>
        <tr>
            <td>--max-new-tokens</td>
            <td>Maximum tokens generated</td>
            <td>512</td>
        </tr>
        <tr>
            <td>--chunk-size</td>
            <td>Document chunk size</td>
            <td>1000</td>
        </tr>
        <tr>
            <td>--chunk-overlap</td>
            <td>Overlap between chunks</td>
            <td>50</td>
        </tr>
    </table>
</div>

<div class="section">
    <h2>📌 Notes</h2>
    <ul>
        <li>Vector store is persistent under <code>vector_store/docs_index/</code></li>
        <li>Chat history is session-based</li>
        <li>No external API required (fully local)</li>
    </ul>
</div>



<div class="section">
    <h2>📜 License</h2>
    <p>This project is licensed under the MIT License.</p>
</div>



<div class="footer">
    <p>© 2026 Rajesh Vhankade. All rights reserved.</p>
</div>

</body>
</html>
