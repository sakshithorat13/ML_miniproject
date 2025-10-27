from flask import Flask, request, jsonify
from flask_cors import CORS
import sys
import os

# Add models directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'models'))

from models import visualization, regression, classification, svm_model, ensemble, nonlinear_regression, clustering, pca_analysis,pca

app = Flask(__name__)
CORS(app)

@app.route("/experiment", methods=["POST"])
def run_experiment():
    try:
        data = request.get_json()
        exp = data.get("experiment")
        
        if exp == "data_visualization":
            return jsonify(visualization.run())
        elif exp == "linear_regression":
            return jsonify(regression.run())
        elif exp == "classification":
            return jsonify(classification.run())
        elif exp == "svm":
            return jsonify(svm_model.run())
        elif exp == "ensemble":
            return jsonify(ensemble.run())
        elif exp == "nonlinear_regression":
            return jsonify(nonlinear_regression.run())
        elif exp == "clustering":
            return jsonify(clustering.run())
        elif exp == "pca":
            return jsonify(pca_analysis.run())
        else:
            return jsonify({"error": "Invalid experiment type"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "Healthcare ML Backend is running!"})

if __name__ == "__main__":
    app.run(debug=True, port=5000)
