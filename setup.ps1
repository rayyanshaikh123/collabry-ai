# This script installs the required Python packages and downloads the spaCy model.

# 1. Install Python packages from requirements.txt
echo "Installing Python packages..."
pip install -r requirements.txt

# 2. Download the spaCy English model
echo "Downloading spaCy model..."
python -m spacy download en_core_web_sm

echo "Setup complete."
