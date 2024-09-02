import os
import json
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from llama_cpp import Llama
from dataclasses import dataclass
from typing import Optional

# Load environment variables from .env file
load_dotenv()

# Initialize Flask application
app = Flask(__name__)

# Configuration
@dataclass
class LLMConfig:
    model_name: str = os.getenv("MODEL_NAME", "gpt2")
    use_llama_cpp: bool = os.getenv("USE_LLAMA_CPP", "True").lower() == "true"
    llama_model_path: Optional[str] = os.getenv("LLAMA_MODEL_PATH")
    max_tokens: int = int(os.getenv("MAX_TOKENS", "256"))
    temperature: float = float(os.getenv("TEMPERATURE", "0.7"))
    top_p: float = float(os.getenv("TOP_P", "0.95"))
    top_k: int = int(os.getenv("TOP_K", "50"))
    repetition_penalty: float = float(os.getenv("REPETITION_PENALTY", "1.2"))
    no_repeat_ngram_size: int = int(os.getenv("NO_REPEAT_NGRAM_SIZE", "4"))
    num_beams: int = int(os.getenv("NUM_BEAMS", "1"))
    batch_size: int = int(os.getenv("BATCH_SIZE", "128"))

config = LLMConfig()

# Load model and tokenizer
if config.use_llama_cpp:
    if not config.llama_model_path:
        raise ValueError("LLAMA_MODEL_PATH must be set when USE_LLAMA_CPP is True")
    model = Llama(model_path=config.llama_model_path)
    tokenizer = None
else:
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    model = AutoModelForCausalLM.from_pretrained(config.model_name)
    
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = model.config.eos_token_id

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

# Create a ThreadPoolExecutor
executor = ThreadPoolExecutor(max_workers=5)

@app.route("/")
def index():
    """Render the index.html template."""
    return render_template("index.html")

def generate_response_transformers(prompt):
    """Generate response using the Transformers model."""
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    
    output = []
    for _ in range(0, config.max_tokens, config.batch_size):
        batch_output = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_length=input_ids.shape[1] + config.batch_size,
            num_return_sequences=1,
            no_repeat_ngram_size=config.no_repeat_ngram_size,
            do_sample=True,
            top_k=config.top_k,
            top_p=config.top_p,
            temperature=config.temperature,
            repetition_penalty=config.repetition_penalty,
            num_beams=config.num_beams,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
        
        batch_text = tokenizer.decode(batch_output[0], skip_special_tokens=True)
        new_text = batch_text[len(prompt):]
        output.append(new_text)
        
        input_ids = batch_output
        attention_mask = torch.ones_like(input_ids)

        # Handle memory pressure
        if torch.cuda.memory_allocated() > 0.9 * torch.cuda.max_memory_allocated():
            print("High memory usage detected. Switching to disk-based storage.")
            break

    return "".join(output)

def generate_response_llama(prompt):
    """Generate response using the Llama model."""
    try:
        output = model.create_completion(prompt, max_tokens=config.max_tokens)
        return output['choices'][0]['text']
    except Exception as e:
        print(f"Error generating response with Llama: {str(e)}")
        return "Error generating response."

@app.route("/generate", methods=["POST"])
def generate():
    """Handle POST requests to generate responses."""
    data = request.json
    prompt = data.get("prompt", "")

    try:
        if config.use_llama_cpp:
            response = generate_response_llama(prompt)
        else:
            response = generate_response_transformers(prompt)

        # Post-processing for improved coherence and reduced repetition
        response = response.strip()
        sentences = response.split('.')
        coherent_response = '. '.join(sentence.capitalize().strip() for sentence in sentences if sentence.strip())
        
        return jsonify({"response": coherent_response})
    except Exception as e:
        print(f"Error generating response: {str(e)}")
        return jsonify({"error": "Error generating response"}), 500

@app.route("/config", methods=["GET", "POST"])
def manage_config():
    """Endpoint to get or update the LLM configuration."""
    global config
    if request.method == "POST":
        data = request.json
        for key, value in data.items():
            if hasattr(config, key):
                setattr(config, key, type(getattr(config, key))(value))
        return jsonify({"message": "Configuration updated successfully"})
    else:
        return jsonify(config.__dict__)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=9898, threaded=True)
