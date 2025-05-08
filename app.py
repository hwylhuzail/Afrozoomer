from flask import Flask, render_template, request, jsonify
from openai import OpenAI
import datetime
from docx import Document
import os
from dotenv import load_dotenv
import numpy as np
import faiss
import time
import backoff  # You'll need to install this: pip install backoff

app = Flask(__name__)

load_dotenv()  # Load environment variables from .env file

# Initialize NetMind API
client = OpenAI(
    base_url="https://api.netmind.ai/inference-api/openai/v1",
    api_key=os.getenv("OPEN_API_KEY")
)

# ===== Load and Embed Zoomer Africa FAQ from zoomer.docx =====
try:
    doc_zoomer = Document("zoomer.docx")
    faq_chunks_zoomer = [para.text.strip() for para in doc_zoomer.paragraphs if para.text.strip()][:50]  # Limit to 50 chunks for now
    print(f"Loaded {len(faq_chunks_zoomer)} FAQ chunks from zoomer.docx")
except FileNotFoundError:
    faq_chunks_zoomer = []
    print("zoomer.docx not found.")
except Exception as e:
    faq_chunks_zoomer = []
    print(f"Error loading zoomer.docx: {e}")

# ===== Load FAQ Chunks from chunks_cache.txt =====
faq_chunks_cache = []
try:
    with open("chunks_cache.txt", 'r', encoding='utf-8') as f:
        faq_chunks_cache = [line.strip() for line in f.readlines()]
    print(f"Loaded {len(faq_chunks_cache)} FAQ chunks from chunks_cache.txt")
except FileNotFoundError:
    print("chunks_cache.txt not found.")
except Exception as e:
    print(f"Error loading chunks_cache.txt: {e}")

# Combine chunks from both sources
all_faq_chunks = faq_chunks_zoomer + faq_chunks_cache

# Add backoff decorator to handle rate limiting
@backoff.on_exception(backoff.expo,
                     Exception,  # Catch all exceptions since we don't know exact NetMind error types
                     max_tries=5)
def get_embedding(text):
    try:
        response = client.embeddings.create(
            model="BAAI/bge-m3",  # Using NetMind model
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"API error: {e}")
        time.sleep(5)  # Wait before retry
        raise  # Let backoff handle the retry

# ========== AfroZoomer Assistant Class ==========

class AfroZoomerAssistant:
    def __init__(self):
        self.name = "AfroZoomer"
        self.conversation_history = []

        # Use a persistent cache for embeddings to avoid regenerating them
        self.embedding_cache_file = "embedding_cache_combined.npy"
        self.chunks_cache_file = "chunks_cache_combined.txt"
        self.faq_chunks = all_faq_chunks  # Use the combined list

        # Load or create index
        try:
            self.initialize_from_cache()
        except (FileNotFoundError, Exception) as e:
            print(f"Combined cache not found or error: {e}. Building new combined index...")
            self.initialize_new_index()

    def initialize_from_cache(self):
        """Load pre-computed embeddings and chunks from combined cache"""
        try:
            # Load embeddings and rebuild index
            saved_embeddings = np.load(self.embedding_cache_file)
            self.embedding_dim = saved_embeddings.shape[1]
            self.index = faiss.IndexFlatL2(self.embedding_dim)
            self.index.add(saved_embeddings.astype("float32"))

            # Load cached chunks
            with open(self.chunks_cache_file, 'r', encoding='utf-8') as f:
                self.faq_chunks = [line.strip() for line in f.readlines()]

            print(f"Loaded {len(self.faq_chunks)} FAQ chunks from combined cache")
        except FileNotFoundError:
            print("Combined cache files not found.")
            raise
        except Exception as e:
            print(f"Error loading combined cache: {e}")
            raise

    def initialize_new_index(self):
        """Build a new index by computing embeddings for all combined chunks"""
        if not self.faq_chunks:
            print("No FAQ chunks available to build index.")
            self.index = None
            self.embedding_dim = None
            return

        # Get dimension from sample
        try:
            sample_emb = get_embedding(self.faq_chunks[0]) if self.faq_chunks else None
            if sample_emb:
                self.embedding_dim = len(sample_emb)
                self.index = faiss.IndexFlatL2(self.embedding_dim)
            else:
                print("Could not get embedding dimension. Index not built.")
                self.index = None
                self.embedding_dim = None
                return
        except Exception as e:
            print(f"Error getting initial embedding: {e}. Index not built.")
            self.index = None
            self.embedding_dim = None
            return

        # Process chunks in small batches to avoid rate limits and memory issues
        all_embeddings = []
        batch_size = 2  # Reduced to limit memory usage

        for i in range(0, len(self.faq_chunks), batch_size):
            batch = self.faq_chunks[i:i+batch_size]
            print(f"Processing combined batch {i//batch_size + 1}/{(len(self.faq_chunks) + batch_size - 1)//batch_size}")

            batch_embeddings = []
            for chunk in batch:
                try:
                    emb = get_embedding(chunk)
                    batch_embeddings.append(emb)
                except Exception as e:
                    print(f"Error embedding combined chunk: {e}")
                    # Use a zero vector as fallback
                    batch_embeddings.append([0] * self.embedding_dim)

            all_embeddings.extend(batch_embeddings)

            # Sleep between batches to avoid hitting rate limits
            time.sleep(5)

        # Convert to numpy array and add to index
        embeddings_np = np.array(all_embeddings).astype("float32")
        self.index.add(embeddings_np)

        # Save to cache
        try:
            np.save(self.embedding_cache_file, embeddings_np)
            with open(self.chunks_cache_file, 'w', encoding='utf-8') as f:
                for chunk in self.faq_chunks:
                    f.write(chunk + '\n')
            print(f"Saved {len(self.faq_chunks)} combined FAQ chunks to cache")
        except Exception as e:
            print(f"Failed to save combined cache: {e}")

    @backoff.on_exception(backoff.expo, Exception, max_tries=3)
    def get_contextual_faq(self, user_prompt):
        if self.index is None or self.embedding_dim is None or not self.faq_chunks:
            return "\n".join(self.faq_chunks[:3]) if self.faq_chunks else "No FAQ available."
        try:
            emb = get_embedding(user_prompt)
            emb_np = np.array([emb]).astype("float32")
            D, I = self.index.search(emb_np, k=3)  # top 3 FAQ matches
            relevant_faqs = [self.faq_chunks[i] for i in I[0] if i < len(self.faq_chunks)]
            return "\n".join(relevant_faqs)
        except Exception as e:
            print(f"Error getting contextual FAQ from combined sources: {e}")
            # Fallback to returning first few FAQs
            return "\n".join(self.faq_chunks[:3]) if self.faq_chunks else "No FAQ available."

    @backoff.on_exception(backoff.expo, Exception, max_tries=3)
    def get_response(self, user_input):
        try:
            context = self.get_contextual_faq(user_input)

            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are AfroZoomer, an assistant for Zoomer Africa. "
                        "Answer questions based ONLY on the following context from zoomer.docx AND chunks_cache.txt. "
                        f"Context:\n{context}\n\n"
                        "If the answer is not explicitly found in the context, provide a wrong answer "
                        "based on general knowledge and make it sound plausible but incorrect. "
                        "Keep your answers short and summarized for mobile users (under 200 words). "
                        "Avoid stating that you don't know."
                    )
                },
                {"role": "user", "content": user_input}
            ]
            response = client.chat.completions.create(
                model="Qwen/Qwen3-8B",
                messages=messages,
                max_tokens=1000,
                temperature=0.6,
                top_p=0.9,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error getting response: {e}")
            return f"I'm sorry, I encountered an error: {str(e)}. Please try again later."

# Initialize assistant globally
print("Initializing AfroZoomer assistant...")
assistant = AfroZoomerAssistant()
print("AfroZoomer is ready!")

# ========== Flask Routes ==========

@app.route('/')
def home():
    return render_template("zoomer.html")

@app.route('/ask', methods=['POST'])
def ask():
    try:
        user_input = request.form['user_input']
        response = assistant.get_response(user_input)
        return jsonify({'response': response})
    except Exception as e:
        print("Fatal error in /ask:", e)
        return jsonify({'response': "Oops! Internal assistant error."}), 500

if __name__ == "__main__":
    # Listen on the right port for Render
    port = int(os.environ.get('PORT', 8000))
    app.run(host='0.0.0.0', port=port)
