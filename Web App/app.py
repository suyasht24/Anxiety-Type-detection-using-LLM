from flask import Flask, request, jsonify
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from flask_cors import CORS

# Load the trained model
MODEL_PATH = "D:\\Research\\anxiety_detection\\trained_anxiety_model"
tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
model = BertForSequenceClassification.from_pretrained(MODEL_PATH)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Anxiety categories and responses
anxiety_types = {
    "PTSD": "You might be experiencing PTSD. Try mindfulness techniques or seek professional support.",
    "OCD": "It seems like OCD. Try structured routines and breathing exercises to ease anxiety.",
    "GAD": "This looks like Generalized Anxiety Disorder. Try relaxation techniques like meditation and deep breathing.",
    "Social Anxiety": "It seems like Social Anxiety. Expose yourself to small social situations gradually."
}

# Greeting detection
greetings = ["hi", "hello", "hey", "how are you", "good morning", "good evening",
             "hii", "helloo", "heyy", "how are youu", "good morningg", "good eveningg",
             "hii bro", "helloo bro", "heyy bro", "how are youu bro", "good morningg bro", "good eveningg bro"]

greeting_responses = [
    "Hello! 😊 I'm here to assist you regarding your anxiety. Can you share how you're feeling?",
    "Hey there! Tell me what's going on in your mind.",
    "Hi! How can I help you? Share your thoughts and I’ll do my best to find your anxiety type!"
]

# Short or non-informative input triggers
# Load non-informative inputs from file
def load_non_informative_inputs(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        return [line.strip().lower() for line in file.readlines() if line.strip()]

non_informative_inputs = load_non_informative_inputs("non_informative_inputs.txt")


# Initialize app
app = Flask(__name__)
CORS(app)

# Anxiety prediction function
def predict_anxiety(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()

    anxiety_label = list(anxiety_types.keys())[predicted_class]
    return anxiety_label, anxiety_types[anxiety_label]

# API route
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    user_text = data.get("text", "").strip().lower()

    if not user_text:
        return jsonify({"response": "Please type something so I can understand how you feel!"}), 400

    # Respond to greetings
    if user_text in greetings:
        return jsonify({"response": greeting_responses[torch.randint(0, len(greeting_responses), (1,)).item()]})

    # Check for short or unclear inputs
    if len(user_text.split()) < 3 or user_text in non_informative_inputs:
        return jsonify({"response": "Hmm... I couldn't understand that. Please describe your feelings or anxiety symptoms in a complete sentence."})

    # Predict anxiety
    anxiety_label, coping_tip = predict_anxiety(user_text)
    return jsonify({"response": f"I detected: **{anxiety_label}**. {coping_tip}"})

# Run app
if __name__ == "__main__":
    app.run(debug=True)
