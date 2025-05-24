#!/bin/bash
# ChatPDF-AI Open Source Run Script

echo "🚀 Starting ChatPDF-AI Open Source..."
echo "======================================"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found!"
    echo "Please run: ./setup_opensource.sh first"
    exit 1
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Check if requirements are installed
echo "🧪 Checking dependencies..."
python3 -c "
try:
    import streamlit, torch, transformers, sentence_transformers, langchain
    print('✅ All dependencies available')
except ImportError as e:
    print(f'❌ Missing dependency: {e}')
    print('Please run: ./setup_opensource.sh')
    exit(1)
"

if [ $? -ne 0 ]; then
    exit 1
fi

# Display system info
echo "💻 System Information:"
python3 -c "
import torch
print(f'   PyTorch: {torch.__version__}')
print(f'   CUDA Available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'   GPU Devices: {torch.cuda.device_count()}')
    print(f'   GPU Name: {torch.cuda.get_device_name(0)}')
else:
    print('   Using CPU only')
"

echo ""
echo "🌐 Starting Streamlit server..."
echo "📱 The app will open in your browser"
echo "🛑 Press Ctrl+C to stop the server"
echo ""

# Run the open source app
streamlit run app_opensource.py
