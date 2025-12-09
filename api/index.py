from flask import Flask, render_template_string, request, jsonify, Response
import nomic
from nomic import embed
import numpy as np
from groq import Groq
from sklearn.metrics.pairwise import cosine_similarity
import os

app = Flask(__name__)

# API Keys - IMPORTANT: Set these as environment variables in Vercel!
NOMIC_API_KEY = os.getenv('NOMIC_API_KEY')
GROQ_API_KEY = os.getenv('GROQ_API_KEY')

if not NOMIC_API_KEY or not GROQ_API_KEY:
    raise ValueError("Please set NOMIC_API_KEY and GROQ_API_KEY environment variables")

# Login to Nomic
nomic.cli.login(NOMIC_API_KEY)

# Initialize Groq client
client = Groq(api_key=GROQ_API_KEY)

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

# Generate embeddings (initialize once when app starts)
print("Initializing embeddings...")
output = embed.text(
    texts=chunks,
    model='nomic-embed-text-v1.5',
    task_type='search_document',
)
doc_embeddings = np.array(output['embeddings'])
print(f"Embedded {len(chunks)} text chunks")

def find_relevant_context(question, top_k=3):
    """Find most relevant text chunks for the question"""
    query_output = embed.text(
        texts=[question],
        model='nomic-embed-text-v1.5',
        task_type='search_query',
    )
    query_embedding = np.array(query_output['embeddings'])
    
    similarities = cosine_similarity(query_embedding, doc_embeddings)[0]
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    relevant_chunks = [chunks[i] for i in top_indices]
    
    return "\n\n".join(relevant_chunks)

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
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🧠 Neuroscience AI Chatbot</h1>
            <p>Ask questions about neural development and the nervous system</p>
        </div>
        
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

            // Add user message
            addMessage(question, 'user');
            userInput.value = '';
            sendBtn.disabled = true;

            // Add loading indicator
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
                addMessage('Sorry, there was an error processing your request.', 'ai');
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
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.get_json()
    question = data.get('question', '')
    
    if not question:
        return jsonify({'error': 'No question provided'}), 400
    
    def generate():
        # Get relevant context
        context = find_relevant_context(question)
        
        # Create prompt
        prompt = f"""Based on the following neuroscience information, answer the question.

Context:
{context}

Question: {question}

Answer:"""
        
        # Stream response from Groq
        completion = client.chat.completions.create(
            model="moonshotai/kimi-k2-instruct-0905",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.6,
            max_completion_tokens=4096,
            top_p=1,
            stream=True,
            stop=None
        )
        
        for chunk in completion:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
    
    return Response(generate(), mimetype='text/plain')

@app.route('/api/info')
def info():
    return jsonify({
        'app': 'Neuroscience RAG Chatbot',
        'embeddings': 'Nomic AI (nomic-embed-text-v1.5)',
        'llm': 'Groq (Moonshot Kimi)',
        'chunks': len(chunks),
        'framework': 'Flask + RAG'
    })

if __name__ == '__main__':
    app.run(debug=True)
