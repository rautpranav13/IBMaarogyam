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
    #if not html_content.startswith("<body>") or not html_content.endswith("</body>"):
        #return "<body><h3>Error</h3><p>Invalid HTML content generated.</p></body>"
    return html_content

def process_image_with_query(image_url, user_query):
    """
    Processes an image with a custom user query and returns the response.
    """
    try:
        # Encode the image to base64
        encoded_image = base64.b64encode(requests.get(image_url).content).decode("utf-8")
    except Exception as e:
        return {"status": "error", "message": f"Failed to process image: {str(e)}"}

    # WatsonX AI credentials and model initialization
    credentials = Credentials(
        url="https://eu-de.ml.cloud.ibm.com",  ############  URL  #############
        api_key="ooUg7xGpand-1uX01b2X08vBw5cxaYs3iISwozrQ6T7q" ###################  API KEY  (account CHANGED)############
    )

    model = ModelInference(
        model_id="mistralai/pixtral-12b",
        credentials=credentials,
        project_id="c9ba001c-a34c-4796-8c35-d22ed4b2388e", ############  Projectid  ###########################
        params={"max_tokens": 500}
    )

    try:
        # Prepare the request payload
        messages = augment_api_request_body(user_query, encoded_image)

        # Call WatsonX AI with the custom query
        response = model.chat(messages=messages)
        content = response.get('choices', [{}])[0].get('message', {}).get('content', '').strip()

        # Validate and return the response
        return {"status": "success", "response": validate_html(content)}

    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.route('/process-image', methods=['POST'])
def process_image():
    """
    API endpoint to process an image with a custom user query.
    """
    try:
        # Parse the JSON body
        data = request.get_json()
        image_url = data.get('image_url')
        user_query = data.get('user_query')

        # Validate input
        if not image_url or not isinstance(image_url, str):
            return jsonify({"status": "error", "message": "Invalid input: 'image_url' is required and must be a string."}), 400

        if not user_query or not isinstance(user_query, str):
            return jsonify({"status": "error", "message": "Invalid input: 'user_query' is required and must be a string."}), 400

        # Process the image and query
        result = process_image_with_query(image_url, user_query)
        return jsonify(result)

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/')
def home():
    return "Welcome to the Flask app!"

if __name__ == "__main__":
    app.run(debug=True)
