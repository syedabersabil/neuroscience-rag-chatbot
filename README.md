# 🧠 Neuroscience RAG Chatbot

An intelligent chatbot powered by Retrieval-Augmented Generation (RAG) that answers questions about neuroscience, neural development, and the nervous system.

## Features

- **RAG Architecture**: Uses Nomic AI embeddings for semantic search
- **Streaming Responses**: Real-time AI responses powered by Groq
- **Beautiful UI**: Modern chat interface with gradient design
- **Serverless**: Deploys as serverless functions on Vercel
- **Fast**: Vector similarity search for instant context retrieval

## Tech Stack

- **Backend**: Python Flask
- **Embeddings**: Nomic AI (nomic-embed-text-v1.5)
- **LLM**: Groq (Moonshot Kimi-K2)
- **Vector Search**: scikit-learn cosine similarity
- **Deployment**: Vercel serverless functions

## Local Setup

1. Clone the repository:
```bash
git clone https://github.com/syedabersabil/neuroscience-rag-chatbot.git
cd neuroscience-rag-chatbot
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set environment variables:
```bash
export NOMIC_API_KEY="your_nomic_api_key"
export GROQ_API_KEY="your_groq_api_key"
```

4. Run the app:
```bash
python api/index.py
```

5. Open http://localhost:5000 in your browser

## Deploy on Vercel

1. Go to [vercel.com](https://vercel.com) and sign in
2. Click **"Add New Project"**
3. Import **syedabersabil/neuroscience-rag-chatbot**
4. Add environment variables:
   - `NOMIC_API_KEY` = your Nomic API key
   - `GROQ_API_KEY` = your Groq API key
5. Click **Deploy**

## How It Works

1. **Embedding Generation**: The neuroscience text is split into chunks and embedded using Nomic AI
2. **Semantic Search**: User questions are embedded and matched against the knowledge base using cosine similarity
3. **Context Retrieval**: Top 3 most relevant chunks are retrieved
4. **LLM Generation**: Groq LLM generates answers based on the retrieved context
5. **Streaming**: Responses are streamed in real-time to the UI

## Example Questions

- What is neurulation?
- How do growth cones work?
- Explain synaptogenesis
- What are critical periods in development?
- What is the role of neuronal death?

## API Endpoints

- `GET /` - Chat interface
- `POST /api/chat` - Send question, get streamed response
- `GET /api/info` - App information

## License

MIT
