from flask import Flask, request, jsonify
from transformers import pipeline
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration,GenerationConfig

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Initialize the Flask app
app = Flask(__name__)
#______________________________  mt5  ________________________________
model_name = "persiannlp/mt5-small-parsinlu-translation_en_fa"
tokenizer = T5Tokenizer.from_pretrained(model_name)
translator = T5ForConditionalGeneration.from_pretrained('./t5_translator1')



def translate_text(text):
    translator.eval()
    input_ids = tokenizer(text, return_tensors="pt").input_ids
    outputs = translator.generate(input_ids,max_new_tokens=500)
    decoded_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return decoded_text



# Load the translation model
#translator = pipeline("translation_en_to_fa", model="persiannlp/mt5-small-parsinlu-translation_en_fa")

# Define a route for the translation API
@app.route('/translate', methods=['POST'])
def translate():
    try:
        # Get the English text from the request body (JSON format)
        data = request.json
        english_text = data.get('text', '')

        if not english_text:
            return jsonify({"error": "No text provided"}), 400

        # Perform translation
        #translation = translator(english_text)[0]['translation_text']

        translation=translate_text(english_text)
        print(translation)



        # Return the translated text as a JSON response
        return jsonify({"translated_text": translation}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run the app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)