import os
from flask import Flask, request, jsonify
import requests
import base64
from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import ModelInference

app = Flask(__name__)

def augment_api_request_body(user_query, images):
    """
    Prepares the message payload for the WatsonX API request.
    """
    image_payload = []
    for image in images:
        image_payload.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{image}",
            }
        })
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": f"You are a helpful assistant. Dont add extra symbols. Please answer the following in 1-2 sentences: {user_query}"},
                *image_payload
            ]
        }
    ]
    return messages

def process_images_and_query(image_urls):
    """
    Processes the images and the user queries for insights and drug schedules, returning a structured response.
    """
    # Encode images in base64
    encoded_images = []
    for url in image_urls:
        try:
            encoded_images.append(base64.b64encode(requests.get(url).content).decode("utf-8"))
        except Exception as e:
            return {"status": "error", "message": f"Failed to fetch or process image at {url}: {str(e)}"}

    # Fetch credentials from environment variables
    api_key = os.getenv("IBM_WATSON_API_KEY")
    url = os.getenv("IBM_WATSON_URL")
    
    # Check if the credentials are set
    if not api_key or not url:
        return {"status": "error", "message": "IBM Watson credentials are missing."}
    
    # Instantiate the model
    credentials = Credentials(
        url=url,
        api_key=api_key
    )

    model = ModelInference(
        model_id="mistralai/pixtral-12b",
        credentials=credentials,
        project_id="09c41291-0ace-4742-8066-b5b2df2d2db0",
        params={
            "max_tokens": 1000
        }
    )

    responses = []
    # Analyze each image for insights and drug schedule
    for i, encoded_image in enumerate(encoded_images):
        try:
            # Insights Query
            insights_query = "Provide insights for the given image in simple language."
            insights_messages = augment_api_request_body(insights_query, [encoded_image])
            insights_response = model.chat(messages=insights_messages)
            insights_content = insights_response['choices'][0]['message']['content']

            # Drug Schedule Query
            schedule_query = "Extract the drug schedule including drug name, quantity, and timing from the given prescription."
            schedule_messages = augment_api_request_body(schedule_query, [encoded_image])
            schedule_response = model.chat(messages=schedule_messages)
            schedule_content = schedule_response['choices'][0]['message']['content']

            responses.append({
                "image_index": i + 1,
                "insights": insights_content,
                "drug_schedule": schedule_content
            })
        except Exception as e:
            responses.append({"image_index": i + 1, "error": str(e)})

    return {
        "status": "success",
        "responses": responses
    }

@app.route('/process-images', methods=['POST'])
def process_images():
    """
    API endpoint to process images and extract insights and drug schedules.
    """
    try:
        data = request.get_json()
        image_urls = data.get('image_urls', [])

        if not image_urls or not isinstance(image_urls, list):
            return jsonify({"status": "error", "message": "Invalid or missing 'image_urls' parameter."}), 400

        result = process_images_and_query(image_urls)
        return jsonify(result)

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

# For Netlify to work, you need a handler function to wrap your Flask app
def handler(event, context):
    from flask_lambda import FlaskLambda
    app_lambda = FlaskLambda(app)
    return app_lambda.handler(event, context)

if __name__ == '__main__':
    app.run(debug=True)
