from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend

# Import your classes - adjust the import path as needed
# Assuming your classes are in a file called 'model_analysis.py' in the same directory
from scanner import GenerativeModelOutputs, EmbeddingOutputs, Attention

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend integration

# Global model instances (initialized lazily)
models = {}


def get_generative_model():
    """Get or create a generative model instance (always GPT-2)"""
    if "gen_gpt2" not in models:
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        model = AutoModelForCausalLM.from_pretrained("gpt2")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        models["gen_gpt2"] = GenerativeModelOutputs(model, tokenizer)
    return models["gen_gpt2"]


def get_embedding_model(model_name="bert-base-uncased"):
    """Get or create an embedding model instance"""
    if f"emb_{model_name}" not in models:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        models[f"emb_{model_name}"] = EmbeddingOutputs(model, tokenizer)
    return models[f"emb_{model_name}"]


def get_attention_model(model_name="gpt2", gen=True):
    """Get or create an attention model instance"""
    key = f"att_{model_name}_{gen}"
    if key not in models:
        models[key] = Attention(model_name, gen)
    return models[key]


@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "message": "Model Analysis API is running"})


@app.route("/models/available", methods=["GET"])
def available_models():
    """Get list of available models"""
    return jsonify(
        {
            "generative_models": ["gpt2"],
            "embedding_models": ["bert-base-uncased", "distilbert-base-uncased"],
            "attention_models": ["gpt2", "bert-base-uncased"],
        }
    )


@app.route("/analyze/sentence-likelihood", methods=["POST"])
def sentence_likelihood():
    """Analyze sentence log likelihoods"""
    try:
        data = request.get_json()
        text = data.get("text", "")

        if not text:
            return jsonify({"error": "Text is required"}), 400

        model = get_generative_model()
        output = model.sentence_log_likelihoods(text)

        return jsonify(
            {
                "tokens": output.get_tokens(),
                "total_likelihood": output.get_total(),
                "perplexity": output.get_perplexity(),
                "num_tokens": len(output.likelihoods_tokenwise),
                "model": "gpt2",
            }
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/analyze/top-k-tokens", methods=["POST"])
def top_k_tokens():
    """Get top-k next token predictions"""
    try:
        data = request.get_json()
        text = data.get("text", "")
        k = data.get("k", 5)
        include_plot = data.get("include_plot", False)

        if not text:
            return jsonify({"error": "Text is required"}), 400

        model = get_generative_model()
        result = model.view_topk(text, k, get_plot=include_plot)

        response = {"top_tokens": result, "input_text": text, "k": k, "model": "gpt2"}

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/analyze/embeddings", methods=["POST"])
def get_embeddings():
    """Get word embeddings"""
    try:
        data = request.get_json()
        words = data.get("words", [])
        model_name = data.get("model", "bert-base-uncased")

        if not words or not isinstance(words, list):
            return jsonify({"error": "Words list is required"}), 400

        model = get_embedding_model(model_name)
        embeddings = model.get_embeddings_output(words)

        # Convert numpy arrays to lists for JSON serialization
        embeddings_dict = {}
        for word, embedding in embeddings.items():
            if hasattr(embedding, "tolist"):
                embeddings_dict[word] = embedding.tolist()
            else:
                embeddings_dict[word] = embedding

        return jsonify(
            {"embeddings": embeddings_dict, "words": words, "model": model_name}
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/analyze/attention", methods=["POST"])
def analyze_attention():
    """Analyze attention scores"""
    try:
        data = request.get_json()
        sentence = data.get("sentence", "")
        model_name = data.get("model", "gpt2")
        is_generative = data.get("generative", True)

        if not sentence:
            return jsonify({"error": "Sentence is required"}), 400

        attention_model = get_attention_model(model_name, is_generative)
        attention_scores = attention_model.attention_scores(sentence)

        # Sort by attention score (descending)
        sorted_scores = sorted(
            attention_scores.items(), key=lambda x: x[1], reverse=True
        )

        response = {
            "attention_scores": attention_scores,
            "sorted_scores": sorted_scores,
            "sentence": sentence,
            "model": model_name,
            "generative": is_generative,
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/analyze/attention/visualize", methods=["POST"])
def visualize_attention():
    """Generate attention visualization"""
    try:
        data = request.get_json()
        sentence = data.get("sentence", "")
        model_name = data.get("model", "gpt2")
        is_generative = data.get("generative", True)
        show_graph = data.get("show_graph", True)

        if not sentence:
            return jsonify({"error": "Sentence is required"}), 400

        attention_model = get_attention_model(model_name, is_generative)

        # Capture the output of view_attention method
        import sys
        from io import StringIO

        old_stdout = sys.stdout
        sys.stdout = captured_output = StringIO()

        attention_model.view_attention(sentence, graph=show_graph)

        sys.stdout = old_stdout
        attention_output = captured_output.getvalue()

        return jsonify(
            {
                "sentence": sentence,
                "model": model_name,
                "attention_output": attention_output,
                "graph_generated": show_graph,
            }
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print("Starting Model Analysis API...")
    print("Available endpoints:")
    print("  GET  /health - Health check")
    print("  GET  /models/available - List available models")
    print("  POST /analyze/sentence-likelihood - Analyze sentence likelihood")
    print("  POST /analyze/top-k-tokens - Get top-k token predictions")
    print("  POST /analyze/embeddings - Get word embeddings")
    print("  POST /analyze/attention - Analyze attention scores")
    print("  POST /analyze/attention/visualize - Generate attention visualization")

    app.run(debug=False, host="0.0.0.0", port=5000)
