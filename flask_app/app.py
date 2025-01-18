#API service
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
    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "You are a highly advanced AI assistant designed to process and analyze medical images. "
                        "Bullet points with short descriptions are preferred. Return the response as minimal "
                        "HTML <body> content, structured with headings, bullet points, and bolded critical information. "
                        "Ensure the output is plain text, concise, and suitable for mobile app display. "
                        f"{user_query}"
                    )
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image}"}
                }
            ]
        }
    ]

def validate_html(html_content):
    """
    Validates if the HTML content starts with <body> and ends with </body>.
    If invalid, returns fallback HTML content.
    """
    if not html_content.startswith("<body>") or not html_content.endswith("</body>"):
        return "<body><h3>Error</h3><p>Invalid HTML content generated.</p></body>"
    return html_content

def process_image_and_query(image_url):
    """
    Processes a single image URL for insights and drug schedules.
    """
    try:
        # Encode the image to base64
        encoded_image = base64.b64encode(requests.get(image_url).content).decode("utf-8")
    except Exception as e:
        return {"status": "error", "message": f"Failed to process image: {str(e)}"}

    # WatsonX AI credentials and model initialization
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
        # Define queries and responses
        queries = {
            "insights": "Extract and summarize the key details present in the image with significant observations.",
            "drug_schedule": "Extract the list of prescribed medicines along with their dosage, frequency, and any specific instructions."
        }

        responses = {}

        # Process each query
        for key, query in queries.items():
            messages = augment_api_request_body(query, encoded_image)
            response = model.chat(messages=messages)
            content = response.get('choices', [{}])[0].get('message', {}).get('content', '').strip()
            responses[key] = validate_html(content)

        return responses

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
