from flask_lambda import FlaskLambda
from flask import Flask, request, jsonify
import requests
import base64
from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import ModelInference

app = FlaskLambda(__name__)

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
                {"type": "text", "text": f"You are a helpful assistant. Answer briefly: {user_query}"},
                *image_payload
            ]
        }
    ]
    return messages

def process_images_and_query(image_urls):
    """
    Processes images and queries for insights and drug schedules.
    """
    encoded_images = []
    for url in image_urls:
        try:
            encoded_images.append(base64.b64encode(requests.get(url).content).decode("utf-8"))
        except Exception as e:
            return {"status": "error", "message": f"Failed to process image: {str(e)}"}

    credentials = Credentials(
        url="https://eu-gb.ml.cloud.ibm.com",
        api_key="your_ibm_api_key"
    )

    model = ModelInference(
        model_id="mistralai/pixtral-12b",
        credentials=credentials,
        project_id="your_project_id",
        params={"max_tokens": 1000}
    )

    responses = []
    for i, encoded_image in enumerate(encoded_images):
        try:
            insights_query = "Provide insights for the given image."
            insights_messages = augment_api_request_body(insights_query, [encoded_image])
            insights_response = model.chat(messages=insights_messages)
            insights_content = insights_response['choices'][0]['message']['content']

            schedule_query = "Extract drug schedules from the image."
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

    return {"status": "success", "responses": responses}

@app.route('/process-images', methods=['POST'])
def process_images():
    """
    API endpoint to process images.
    """
    try:
        data = request.get_json()
        image_urls = data.get('image_urls', [])

        if not image_urls or not isinstance(image_urls, list):
            return jsonify({"status": "error", "message": "Invalid 'image_urls'."}), 400

        result = process_images_and_query(image_urls)
        return jsonify(result)

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/')
def home():
    return "Welcome to the Flask app on Netlify!"

if __name__ == "__main__":
    app.run(debug=True)
