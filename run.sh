#!/bin/bash

# Остановить выполнение при ошибке
set -e

echo "🔧 Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

echo "📦 Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo "🚀 Running inference script..."
python run_inference2.py

echo "✅ Done. Result saved to result.csv"
