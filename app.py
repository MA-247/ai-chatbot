from flask import Flask, request, jsonify, send_from_directory
from transformers import GPT2Tokenizer, GPT2LMHeadModel, pipeline

app = Flask(__name__)

# Load your fine-tuned model
model_name = 'fine-tuned-gpt2'  # Update with your model path
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

chatbot = pipeline('text-generation', model=model, tokenizer=tokenizer)

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get('message')
    response = chatbot(user_input)[0]['generated_text']
    return jsonify({'response': response})

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

if __name__ == '__main__':
    app.run(debug=True)
