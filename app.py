from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer, util
import torch
import os
app = Flask(__name__)

# Load model
model = SentenceTransformer("sentence-transformers/paraphrase-MiniLM-L3-v2")

@app.route('/similarity', methods=['POST'])
def compute_similarity():
    data = request.json
    text1 = data.get("text1", "")
    text2 = data.get("text2", "")

    # Compute embeddings
    embedding1 = model.encode(text1, convert_to_tensor=True)
    embedding2 = model.encode(text2, convert_to_tensor=True)

    # Compute cosine similarity
    similarity_score = util.pytorch_cos_sim(embedding1, embedding2).item()

    # Convert similarity from [-1,1] to [0,1]
    similarity_score = (similarity_score + 1) / 2

    return jsonify({"similarity score": similarity_score})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))  # Ensure Render detects open port
    app.run(host='0.0.0.0', port=port)  # Remove debug=True
    

