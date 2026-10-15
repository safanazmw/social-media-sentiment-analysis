# Social Media Sentiment Analysis

A Flask web application for exploring the supplied social-media dataset and predicting sentiment for text entered by a user. The UI uses a glassmorphism style with **light navy blue and off-beige** colors.

## What it includes
- Dashboard with dataset counts and sentiment distribution.
- Dataset search and filters by entity and sentiment.
- Single-post sentiment prediction.
- TF-IDF text features and Logistic Regression classifier.
- Four labels from the dataset: Positive, Negative, Neutral, Irrelevant.
- Model evaluation metrics saved locally after training.

## Requirements
- Python 3.10–3.12 recommended.
- `Data/dataset.csv` included in this project.
- No MySQL Workbench, database password, or external API key is required.

## Run on Windows (VS Code)
1. Extract this ZIP and open the extracted folder in VS Code. The folder you open must contain `app.py`.
2. Open **Terminal → New Terminal**.
3. Create a virtual environment:
   ```powershell
   py -m venv .venv
   ```
4. Activate it:
   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```
   If activation is blocked, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` in that terminal and activate again.
5. Install dependencies:
   ```powershell
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```
6. Train the model (first run; may take a few minutes):
   ```powershell
   python -m Analysis.train_model
   ```
7. Start the local website:
   ```powershell
   python app.py
   ```
8. Open the URL printed in the terminal, normally `http://127.0.0.1:5000`.

## Secrets and deployment
- For local use, no secret or API key is needed. For public deployment, set `FLASK_SECRET_KEY` to a long random value in the hosting provider's environment settings. Never commit a real `.env` file or secret to GitHub.
- `python app.py` is for local development. For production, use Waitress, for example `waitress-serve --listen=0.0.0.0:8080 app:app`, and set `FLASK_DEBUG=0`.

## Notes
- The app uses the supplied CSV and does not insert fake/demo dataset records.
- Model accuracy is measured on a held-out test split and does not guarantee correct predictions on every new post.
- The dataset has no date column, so this app does not claim to show sentiment trends over time.
