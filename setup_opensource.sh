#!/bin/bash
# ChatPDF-AI Open Source Setup Script
# Completely local setup - no API keys required!

echo "🚀 ChatPDF-AI Open Source Setup"
echo "================================"

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

echo "✅ Python 3 found: $(python3 --version)"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
    echo "✅ Virtual environment created"
else
    echo "✅ Virtual environment already exists"
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# Install open source requirements
echo "📚 Installing open source dependencies..."
echo "This may take a few minutes as models need to be downloaded..."
pip install -r requirements_opensource.txt

# Check installation
echo "🧪 Testing installation..."
python3 -c "
import streamlit
import torch
import transformers
import sentence_transformers
import langchain
import langchain_community
print('✅ All dependencies installed successfully!')
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'Streamlit version: {streamlit.__version__}')
print(f'Transformers version: {transformers.__version__}')
"

echo ""
echo "🎉 Setup Complete!"
echo "=================="
echo ""
echo "🏃‍♂️ To run the open source version:"
echo "   ./run_opensource.sh"
echo ""
echo "🌐 Or manually:"
echo "   source venv/bin/activate"
echo "   streamlit run app_opensource.py"
echo ""
echo "📝 Features:"
echo "   ✅ Completely local processing"
echo "   ✅ No API keys required"
echo "   ✅ Privacy-focused"
echo "   ✅ Works offline"
echo ""
echo "💡 Note: First run will download ML models (~500MB)"
echo "🚀 GPU recommended for better performance"
