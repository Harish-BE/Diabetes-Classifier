from flask import Flask, jsonify, request, render_template, send_file
import pandas as pd
import pickle

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    if not file.filename.endswith('.csv'):
        return jsonify({"error": "Invalid file type. Only CSV files are accepted."}), 400

    df= pd.read_csv(file)

    with open('model.pkl', 'rb') as f:
        loaded_model = pickle.load(f)

    predictions = loaded_model.predict(df)
    df['Predictions'] = predictions

    output_file_path = "predicted_file.csv"
    df.to_csv(output_file_path, index=False)

    return send_file(output_file_path, as_attachment=True)
    
# Run the app
if __name__ == '__main__':
    app.run(debug=True)
