#!/bin/bash

# VERA Setup Script
# This script downloads all required models and dependencies for the VERA system

set -e  # Exit on any error

echo "🚀 VERA Setup Script"
echo "===================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if conda is available
if ! command -v conda &> /dev/null; then
    print_error "Conda is not installed or not in PATH"
    print_error "Please install Miniconda or Anaconda first:"
    print_error "https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

print_status "Conda found: $(conda --version)"

# Check if we're in the right directory
if [ ! -f "backend/app.py" ]; then
    print_error "Please run this script from the VERA root directory"
    print_error "Expected to find backend/app.py in current directory"
    exit 1
fi

print_status "Running from VERA root directory ✓"

# Create necessary directories
print_status "Creating necessary directories..."
mkdir -p backend/models/piper
mkdir -p backend/piper
mkdir -p data/sessions
mkdir -p logs

# Step 1: Create conda environment
print_status "Creating conda environment 'vera'..."
if conda env list | grep -q "vera"; then
    print_warning "Conda environment 'vera' already exists"
    read -p "Do you want to recreate it? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_status "Removing existing 'vera' environment..."
        conda env remove -n vera -y
        print_status "Creating new 'vera' environment..."
        conda create -n vera python=3.11 -y
    else
        print_status "Using existing 'vera' environment"
    fi
else
    print_status "Creating new 'vera' environment..."
    conda create -n vera python=3.11 -y
fi

# Step 2: Activate environment and install system dependencies
print_status "Activating conda environment and installing system dependencies..."
eval "$(conda shell.bash hook)"
conda activate vera

# Install system dependencies
print_status "Installing system dependencies via conda..."
conda install -c conda-forge -y ffmpeg

# Install pip if not available
if ! command -v pip &> /dev/null; then
    print_status "Installing pip..."
    conda install pip -y
fi

# Step 3: Install Python dependencies
print_status "Installing Python dependencies..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
else
    print_status "Installing core Python packages..."
    pip install fastapi uvicorn websockets faster-whisper ctranslate2 torch torchaudio numpy scipy pyyaml soundfile requests
fi

# Step 4: Download Piper TTS binary and voices
print_status "Downloading Piper TTS binary..."
cd backend

# Download Piper binary (Linux x86_64)
PIPER_VERSION="1.2.0"
PIPER_URL="https://github.com/rhasspy/piper/releases/download/v${PIPER_VERSION}/piper_${PIPER_VERSION}_linux.tar.gz"

if [ ! -f "piper/piper" ]; then
    print_status "Downloading Piper TTS binary from ${PIPER_URL}..."
    wget -O piper.tar.gz "${PIPER_URL}"
    tar -xzf piper.tar.gz
    rm piper.tar.gz
    print_success "Piper TTS binary downloaded and extracted"
else
    print_success "Piper TTS binary already exists"
fi

# Make piper executable
chmod +x piper/piper

# Step 5: Download Piper voices
print_status "Downloading Piper voices..."
cd models/piper

# Voice URLs (HuggingFace)
VOICES=(
    "en_US-amy-medium"
    "en_US-lessac-medium" 
    "en_US-ryan-high"
)

for voice in "${VOICES[@]}"; do
    if [ ! -f "${voice}.onnx" ]; then
        print_status "Downloading voice: ${voice}"
        
        # Download .onnx model
        wget -O "${voice}.onnx" "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/${voice}/${voice}.onnx"
        
        # Download .onnx.json config
        wget -O "${voice}.onnx.json" "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/${voice}/${voice}.onnx.json"
        
        print_success "Downloaded voice: ${voice}"
    else
        print_success "Voice already exists: ${voice}"
    fi
done

cd ../..

# Step 6: Download Whisper models (this will happen on first run)
print_status "Whisper models will be downloaded automatically on first run"
print_status "This includes the large-v3 model (~3GB)"

# Step 7: Set up logging
print_status "Setting up logging directory..."
mkdir -p logs
touch logs/vera.log

# Step 8: Create a simple test script
print_status "Creating test script..."
cat > test_setup.py << 'EOF'
#!/usr/bin/env python3
"""
Simple test script to verify VERA setup
"""
import sys
import os

# Add backend directory to Python path
backend_path = os.path.join(os.path.dirname(__file__), 'backend')
sys.path.insert(0, backend_path)

try:
    # Test imports with correct class names
    from asr import ASR
    from tts import TTS
    from vad import StreamingVAD
    print("✅ All imports successful")
    
    # Test VAD (simple initialization)
    vad = StreamingVAD()
    print("✅ VAD initialized")
    
    # Test TTS (without loading models - just check class)
    print("✅ TTS class available")
    
    print("\n🎉 VERA setup verification successful!")
    print("You can now run: cd backend && python app.py")
    
except Exception as e:
    print(f"❌ Setup verification failed: {e}")
    print("This might be due to missing dependencies or Python path issues.")
    print("Try running: cd backend && python -c 'import asr, tts, vad'")
    sys.exit(1)
EOF

chmod +x test_setup.py

# Step 9: Final verification
print_status "Running setup verification..."
python test_setup.py

print_success "VERA setup completed successfully!"
echo ""
print_status "Next steps:"
echo "1. cd backend"
echo "2. eval \"\$(conda shell.bash hook)\""
echo "3. conda activate vera"
echo "4. python app.py"
echo ""
print_status "The application will be available at: http://localhost:8000"
echo ""
print_warning "Note: First run will download Whisper models (~3GB) and may take a few minutes"
