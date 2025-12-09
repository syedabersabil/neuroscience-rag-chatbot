from flask import Flask, render_template_string, request, jsonify, Response
import os
import re
from collections import Counter
import math

app = Flask(__name__)

# API Key - with fallback for testing
GROQ_API_KEY = os.getenv('GROQ_API_KEY', '')

# Only import and initialize Groq if key is available
if GROQ_API_KEY:
    try:
        from groq import Groq
        client = Groq(api_key=GROQ_API_KEY)
    except Exception as e:
        print(f"Error initializing Groq: {e}")
        client = None
else:
    client = None
    print("Warning: GROQ_API_KEY not set")

# Neuroscience knowledge base
neuroscience_text = """Neuronal growth cones in the spinal cord of a chick embryo. These hand-like structures, on the tips of developing axons, carry out the most amazing feat of neural development: its wiring. Courtesy of the Cajal Institute, "Cajal Legacy," Spanish National Research Council (CSIC), Madrid, Spain.

After a short period spent in Brussels as a guest of a neurological institute, I returned to Turin on the verge of the invasion of Belgium by the German army, Spring 1940, to join my family. The two alternatives left then to us were, either to emigrate to the United States, or to pursue some activity that needed neither support nor connection with the outside Aryan world where we lived. My family chose this second alternative. I then decided to build a small research unit at home and installed it in my bedroom. My inspiration was a 1934 article by Viktor Hamburger reporting on the effects of limb extirpation in chick embryos. My project had barely started when Giuseppe Levi, who had escaped from Belgium invaded by Nazis, returned to Turin and joined me, thus becoming, to my great pride, my first and only assistant. The heavy bombing of Turin by Anglo American air forces in 1941 made it imperative to abandon Turin and move to a country cottage where I rebuilt my mini-laboratory and resumed my experiments.
—Rita Levi-Montalcini, Nobel Lecture, 1986

The Nervous System

The nervous system is divided into the central nervous system (CNS) and the peripheral nervous system (PNS). The CNS is itself subdivided into the brain and spinal cord, connected and protected by our skull and spine.

The brain itself is subdivided into regions, grouped, from top to bottom, into forebrain, midbrain, and hindbrain, which then leads to the spinal cord. The forebrain has two cerebral hemispheres that, in humans, balloon into an enormous cerebral cortex.

How the Nervous System Develops

Remarkably, the brain actually assembles itself. The nervous system develops through a stereotypical set of stages. It follows a step-by-step temporal logic of causation whereby what happens at a given time determines what happens next.

Neurulation and Patterning

The first step in brain development is the formation of the neural tube. This process is called neurulation. The CNS starts as a tubular infolding from the ectoderm, the outermost layer of the embryo.

Wiring and Pathfinding

Axonal pathfinding is done by a specialized structure at the tip of the growing axons, the growth cone. Growth cones are amazing navigators. For example, some neurons from our motor cortex send an axon to the lower part of the spinal cord.

Synaptogenesis

Synaptogenesis is the formation of synaptic connections. A synapse is formed when a growth cone finds the body or dendrites of its target neuron.

Neuronal Death

Massive cell death of neurons occurs during development. Hamburger argued that nature used this culling technique to match the number of spinal cord motoneurons to the number of muscles.

Critical Periods

Activity-dependent refinement happens at particular times during development. Torsten Wiesel and David Hubel discovered that a brief period of visual deprivation during development forever altered neural circuits in the visual cortex."""

# Split text into chunks
chunks = [chunk.strip() for chunk in neuroscience_text.split('\n\n') if chunk.strip()]

# Simple tokenizer
def tokenize(text):
    """Convert text to lowercase tokens"""
    return re.findall(r'\w+', text.lower())

# Build simple TF-IDF index
def compute_tf(tokens):
    """Compute term frequency"""
    if not tokens:
        return {}
    counter = Counter(tokens)
    total = len(tokens)
    return {word: count / total for word, count in counter.items()}

def compute_idf(chunks):
    """Compute inverse document frequency"""
    doc_count = len(chunks)
    word_doc_count = {}
    
    for chunk in chunks:
        words = set(tokenize(chunk))
        for word in words:
            word_doc_count[word] = word_doc_count.get(word, 0) + 1
    
    return {word: math.log(doc_count / count) for word, count in word_doc_count.items()}

# Precompute TF-IDF for all chunks
try:
    print("Building search index...")
    idf = compute_idf(chunks)
    chunk_vectors = []
    for chunk in chunks:
        tokens = tokenize(chunk)
        tf = compute_tf(tokens)
        tfidf = {word: tf[word] * idf.get(word, 0) for word in tf}
        chunk_vectors.append(tfidf)
    print(f"Indexed {len(chunks)} text chunks")
except Exception as e:
    print(f"Error building index: {e}")
    chunk_vectors = []

def cosine_similarity_tfidf(vec1, vec2):
    """Calculate cosine similarity between TF-IDF vectors"""
    try:
        # Get common words
        common = set(vec1.keys()) & set(vec2.keys())
        
        if not common:
            return 0.0
        
        # Dot product
        dot = sum(vec1[w] * vec2[w] for w in common)
        
        # Magnitudes
        mag1 = math.sqrt(sum(v * v for v in vec1.values()))
        mag2 = math.sqrt(sum(v * v for v in vec2.values()))
        
        if mag1 == 0 or mag2 == 0:
            return 0.0
        
        return dot / (mag1 * mag2)
    except Exception as e:
        print(f"Error in similarity: {e}")
        return 0.0

def find_relevant_context(question, top_k=3):
    """Find most relevant text chunks for the question"""
    try:
        # Compute TF-IDF for query
        query_tokens = tokenize(question)
        query_tf = compute_tf(query_tokens)
        query_tfidf = {word: query_tf[word] * idf.get(word, 0) for word in query_tf}
        
        # Calculate similarities
        similarities = []
        for chunk_vec in chunk_vectors:
            sim = cosine_similarity_tfidf(query_tfidf, chunk_vec)
            similarities.append(sim)
        
        # Get top k indices
        indexed_sims = list(enumerate(similarities))
        indexed_sims.sort(key=lambda x: x[1], reverse=True)
        top_indices = [idx for idx, _ in indexed_sims[:top_k]]
        
        relevant_chunks = [chunks[i] for i in top_indices]
        return "\n\n".join(relevant_chunks)
    except Exception as e:
        print(f"Error finding context: {e}")
        return neuroscience_text[:1000]  # Fallback to first 1000 chars

# HTML Template
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Neuroscience AI Chatbot</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
            font-family: 'Segoe UI', system-ui, sans-serif;
        }
        body {
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            min-height: 100vh;
            padding: 20px;
        }
        .container {
            max-width: 900px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }
        .header h1 {
            font-size: 2rem;
            margin-bottom: 10px;
        }
        .header p {
            opacity: 0.9;
            font-size: 1.1rem;
        }
        .chat-container {
            height: 500px;
            overflow-y: auto;
            padding: 20px;
            background: #f8f9fa;
        }
        .message {
            margin-bottom: 15px;
            padding: 15px 20px;
            border-radius: 15px;
            max-width: 80%;
            animation: fadeIn 0.3s;
        }
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        .user-message {
            background: #667eea;
            color: white;
            margin-left: auto;
            text-align: right;
        }
        .ai-message {
            background: white;
            color: #333;
            border: 2px solid #e9ecef;
        }
        .input-container {
            padding: 20px;
            background: white;
            border-top: 2px solid #e9ecef;
            display: flex;
            gap: 10px;
        }
        #userInput {
            flex: 1;
            padding: 15px;
            border: 2px solid #e9ecef;
            border-radius: 25px;
            font-size: 1rem;
            outline: none;
        }
        #userInput:focus {
            border-color: #667eea;
        }
        #sendBtn {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 15px 30px;
            border-radius: 25px;
            font-size: 1rem;
            font-weight: 600;
            cursor: pointer;
            transition: transform 0.2s;
        }
        #sendBtn:hover {
            transform: translateY(-2px);
        }
        #sendBtn:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }
        .loading {
            display: none;
            padding: 15px;
            background: white;
            border-radius: 15px;
            max-width: 80px;
            border: 2px solid #e9ecef;
        }
        .loading.active {
            display: block;
        }
        .dot {
            display: inline-block;
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: #667eea;
            margin: 0 2px;
            animation: bounce 1.4s infinite ease-in-out both;
        }
        .dot:nth-child(1) { animation-delay: -0.32s; }
        .dot:nth-child(2) { animation-delay: -0.16s; }
        @keyframes bounce {
            0%, 80%, 100% { transform: scale(0); }
            40% { transform: scale(1); }
        }
        .info {
            padding: 20px;
            background: #e7f3ff;
            border-left: 4px solid #667eea;
            margin: 20px;
            border-radius: 8px;
        }
        .info h3 {
            color: #667eea;
            margin-bottom: 10px;
        }
        .warning {
            padding: 15px;
            background: #fff3cd;
            border-left: 4px solid #ffc107;
            margin: 20px;
            border-radius: 8px;
            color: #856404;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🧠 Neuroscience AI Chatbot</h1>
            <p>Ask questions about neural development and the nervous system</p>
        </div>
        
        {% if not api_key_set %}
        <div class="warning">
            <strong>⚠️ API Key Missing:</strong> The GROQ_API_KEY environment variable is not set. AI responses will not work until configured in Vercel settings.
        </div>
        {% endif %}
        
        <div class="info">
            <h3>💡 Try asking:</h3>
            <p>• What is neurulation?<br>
            • How do growth cones work?<br>
            • Explain synaptogenesis<br>
            • What are critical periods in development?</p>
        </div>
        
        <div class="chat-container" id="chatContainer">
            <div class="message ai-message">
                Hello! I'm your neuroscience AI assistant. Ask me anything about neural development, the nervous system, or neuroscience concepts! 🔬
            </div>
        </div>
        
        <div class="input-container">
            <input type="text" id="userInput" placeholder="Ask a neuroscience question..." />
            <button id="sendBtn" onclick="sendMessage()">Send</button>
        </div>
    </div>

    <script>
        const chatContainer = document.getElementById('chatContainer');
        const userInput = document.getElementById('userInput');
        const sendBtn = document.getElementById('sendBtn');

        userInput.addEventListener('keypress', function(e) {
            if (e.key === 'Enter') sendMessage();
        });

        async function sendMessage() {
            const question = userInput.value.trim();
            if (!question) return;

            addMessage(question, 'user');
            userInput.value = '';
            sendBtn.disabled = true;

            const loadingDiv = document.createElement('div');
            loadingDiv.className = 'loading active';
            loadingDiv.innerHTML = '<span class="dot"></span><span class="dot"></span><span class="dot"></span>';
            chatContainer.appendChild(loadingDiv);
            chatContainer.scrollTop = chatContainer.scrollHeight;

            try {
                const response = await fetch('/api/chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ question })
                });

                const reader = response.body.getReader();
                const decoder = new TextDecoder();
                let aiResponse = '';

                loadingDiv.remove();
                const aiMessageDiv = document.createElement('div');
                aiMessageDiv.className = 'message ai-message';
                chatContainer.appendChild(aiMessageDiv);

                while (true) {
                    const { value, done } = await reader.read();
                    if (done) break;
                    
                    const chunk = decoder.decode(value);
                    aiResponse += chunk;
                    aiMessageDiv.textContent = aiResponse;
                    chatContainer.scrollTop = chatContainer.scrollHeight;
                }
            } catch (error) {
                loadingDiv.remove();
                addMessage('Sorry, there was an error: ' + error.message, 'ai');
            }

            sendBtn.disabled = false;
        }

        function addMessage(text, type) {
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${type}-message`;
            messageDiv.textContent = text;
            chatContainer.appendChild(messageDiv);
            chatContainer.scrollTop = chatContainer.scrollHeight;
        }
    </script>
</body>
</html>
'''

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE, api_key_set=bool(GROQ_API_KEY))

@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        question = data.get('question', '')
        
        if not question:
            return jsonify({'error': 'No question provided'}), 400
        
        if not client:
            return Response("Error: GROQ_API_KEY environment variable not set. Please configure it in Vercel settings.", mimetype='text/plain')
        
        def generate():
            try:
                # Get relevant context using TF-IDF
                context = find_relevant_context(question)
                
                # Create prompt
                prompt = f"""Based on the following neuroscience information, answer the question.

Context:
{context}

Question: {question}

Answer:"""
                
                # Stream response from Groq
                completion = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.6,
                    max_completion_tokens=2048,
                    top_p=1,
                    stream=True,
                    stop=None
                )
                
                for chunk in completion:
                    if chunk.choices[0].delta.content:
                        yield chunk.choices[0].delta.content
            except Exception as e:
                yield f"Error generating response: {str(e)}"
        
        return Response(generate(), mimetype='text/plain')
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/health')
def health():
    return jsonify({
        'status': 'ok',
        'api_key_set': bool(GROQ_API_KEY),
        'chunks_indexed': len(chunks),
        'vectors_built': len(chunk_vectors)
    })

@app.route('/api/info')
def info():
    return jsonify({
        'app': 'Neuroscience RAG Chatbot',
        'retrieval': 'TF-IDF (keyword-based)',
        'llm': 'Groq (Llama 3.3 70B)',
        'chunks': len(chunks),
        'framework': 'Flask + RAG (ultra-lightweight)',
        'api_configured': bool(GROQ_API_KEY)
    })

if __name__ == '__main__':
    app.run(debug=True)
