# Urban Carbon Emissions Dashboard

An interactive dashboard for the synthetic smart-city carbon-emission risk
classifier in `carbon_urban.py`. The app trains the existing XGBoost model,
lets you adjust district sensor readings, and displays its risk prediction,
class probabilities, and held-out evaluation metrics.

## Run locally on Windows

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py
```

If you do not already have a virtual environment, create one first:

```powershell
py -m venv .venv
```

Streamlit prints a local URL (usually `http://localhost:8501`) to open in your
browser. The first launch trains the model; later reruns reuse the cached model.
