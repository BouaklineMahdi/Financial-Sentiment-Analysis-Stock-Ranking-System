from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_cors import CORS, cross_origin  # Import CORS and cross_origin

app = Flask(__name__)
CORS(app)  # Enable CORS for your Flask app globally

# Define the index route
@app.route("/", methods=['GET', 'POST'])
def index():
    try:
        if request.method == 'GET':
            print('Received GET request')  # Debug print for GET requests
        elif request.method == 'POST':
            print('Received POST request')  # Debug print for POST requests
            # Process the data from the POST request if needed
        return render_template('index.html')
    except Exception as e:
        print(f"Error occurred in index route: {e}")  # Debug print for index route errors

# Define a route to start analysis
@app.route("/start_analysis", methods=['GET'])
@cross_origin()  # Enable CORS specifically for the /start_analysis route
def start_analysis():
    try:
        # Perform analysis logic here
        # For now, let's just return a simple response
        print('Starting analysis')  # Debug print for starting analysis
        return jsonify({'message': 'Analysis started successfully'})
    except Exception as e:
        print(f"Error occurred in start_analysis route: {e}")  # Debug print for start_analysis route errors

# Define a route to serve static files
@app.route('/static/<path:path>')
def send_static(path):
    return send_from_directory('static', path)

if __name__ == '__main__':
    app.run(debug=True)