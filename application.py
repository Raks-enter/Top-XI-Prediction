from flask import Flask, request, render_template
import pandas as pd
from src.pipeline.predict_pipeline import PredictPipeline
import os

application = Flask(__name__)
app = application

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict_top_11():
    try:
        file = request.files.get('file')
        if not file:
            return render_template('index.html', error="Please upload a CSV file.")

        # Read the uploaded CSV file into DataFrame
        input_df = pd.read_csv(file)

        # Run prediction pipeline
        predictor = PredictPipeline()
        top_11_df = predictor.predict(input_df)  # Should return a DataFrame with top 11 players

        # Render result page with top 11 table
        return render_template(
            'result.html',
            tables=[top_11_df.to_html(classes='data', header=True, index=False)],
            titles=top_11_df.columns.values
        )

    except Exception as e:
        return render_template('index.html', error=f"Error: {str(e)}")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
