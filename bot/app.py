from flask import Flask, request, jsonify
from auth_cf import handle_request

app = Flask(__name__)

@app.route('/my-webhook', methods=['POST'])
def webhook():
    req = request.get_json(silent=True, force=True)
    response = process_request(req)
    return jsonify(response)

def process_request(req):
    tag = req['fulfillmentInfo'].get('tag', '')
    parameters = req.get('sessionInfo', {}).get('parameters', {})
    session_id = req.get('sessionInfo', {}).get('session', '').split('/')[-1]
    input_text = req.get('text', '')
    if not input_text:
        input_text = req.get('transcript', '')

    response_text = handle_request(tag, input_text, parameters)

    response = {
        "fulfillment_response": {
            "messages": [
                {
                    "text": {
                        "text": [response_text]
                    }
                }
            ]
        },
        "sessionInfo": {
            "parameters": parameters
        }
    }
    return response

if __name__ == '__main__':
    app.run(port=5000, debug=True)
