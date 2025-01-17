from flask import Flask, request, jsonify
import requests
import base64
from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import ModelInference

app = Flask(__name__)

def augment_api_request_body(user_query, image):
    """
    Prepares the message payload for the WatsonX API request.
    """
    image_payload = [{
        "type": "image_url",
        "image_url": {
            "url": f"data:image/jpeg;base64,{image}",
        }
    }]

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": f"You are a helpful assistant. Answer briefly: {user_query}"},
                *image_payload
            ]
        }
    ]
    return messages

def process_image_and_query(image_url):
    """
    Processes a single image URL for insights and drug schedules.
    """
    try:
        # Encode the image to base64
        encoded_image = base64.b64encode(requests.get(image_url).content).decode("utf-8")
    except Exception as e:
        return {"status": "error", "message": f"Failed to process image: {str(e)}"}

    credentials = Credentials(
        url="https://eu-gb.ml.cloud.ibm.com",
        api_key="6P3ojg1AqGOCjwq-w1XVYxNuup_9Dmqqed9zYH8uTo-r"
    )

    model = ModelInference(
        model_id="mistralai/pixtral-12b",
        credentials=credentials,
        project_id="209d09a2-9ebd-4635-826b-250210a27c4e",
        params={"max_tokens": 200}
    )

    try:
        # Insights query
        insights_query = "Provide insights for the given image."
        insights_messages = augment_api_request_body(insights_query, encoded_image)
        insights_response = model.chat(messages=insights_messages)
        insights_content = insights_response['choices'][0]['message']['content']

        # Drug schedule query
        schedule_query = "Extract drug schedules from the image."
        schedule_messages = augment_api_request_body(schedule_query, encoded_image)
        schedule_response = model.chat(messages=schedule_messages)
        schedule_content = schedule_response['choices'][0]['message']['content']

        return {
            "status": "success",
            "insights": insights_content,
            "drug_schedule": schedule_content
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.route('/process-image', methods=['POST'])
def process_image():
    """
    API endpoint to process a single image.
    """
    try:
        # Get the input data
        data = request.get_json()
        
        # Check if the image_url key is present and it's a string
        image_url = data.get('image_url')
        
        if not image_url or not isinstance(image_url, str):
            return jsonify({"status": "error", "message": "Invalid input, expected a single image URL as a string."}), 400

        # Process the image and get the result
        result = process_image_and_query(image_url)
        return jsonify(result)

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/')
def home():
    return "Welcome to the Flask app!"

if __name__ == "__main__":
    app.run(debug=True)
