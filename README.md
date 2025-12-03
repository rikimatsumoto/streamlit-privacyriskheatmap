# Privacy Risk Heatmap – Smart Camera Facial Embedding Risk

This repo contains a Streamlit app that visualizes **privacy risk** for several popular smart-camera devices
(Ring, Nest, Eufy, Wyze, Arlo) based on:

- Camera **vertical resolution**
- **Vertical field of view (FOV)**
- **Distance** from the camera to a subject
- Research-backed **face pixel thresholds** for facial recognition

The app computes an approximate **face pixel height** for each device–distance combination using a simple
geometric projection formula, then maps that to a qualitative privacy risk level and displays a heatmap.

## Quickstart

```bash
# Create and activate a virtual environment (optional but deff recommended)
python -m venv .venv
source .venv/bin/activate  # on Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app/streamlit_app.py # or try this: python -m streamlit run app/streamlit_app.py
```

## Folder Structure

```text
privacy_risk_heatmap_app/
├─ app/
│  ├─ streamlit_app.py      # Main Streamlit entrypoint
│  └─ data/
├─ src/
├─ tests/
├─ config/
├─ docs/
├─ requirements.txt
├─ .gitignore
└─ README.md
```
