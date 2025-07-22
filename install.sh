#!/bin/bash
# Shvayambhu One-Line Installer
# Usage: curl -sSL https://shvayambhu.ai/install.sh | bash

set -e

echo "🚀 Installing Shvayambhu LLM..."
echo "================================"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check OS
if [[ "$OSTYPE" != "darwin"* ]]; then
    echo -e "${RED}❌ Shvayambhu currently only supports macOS with Apple Silicon${NC}"
    exit 1
fi

# Check Apple Silicon
if [[ $(uname -m) != "arm64" ]]; then
    echo -e "${RED}❌ Shvayambhu requires Apple Silicon (M1/M2/M3/M4)${NC}"
    exit 1
fi

# Check Python
if ! command -v python3 &> /dev/null; then
    echo -e "${YELLOW}⚠️  Python 3 not found. Installing via Homebrew...${NC}"
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    brew install python@3.11
fi

# Check Python version
PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
REQUIRED_VERSION="3.8"
if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then
    echo -e "${RED}❌ Python $REQUIRED_VERSION or higher required (found $PYTHON_VERSION)${NC}"
    exit 1
fi

# Check/Install Homebrew
if ! command -v brew &> /dev/null; then
    echo "📦 Installing Homebrew..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
fi

# Check/Install Ollama
if ! command -v ollama &> /dev/null; then
    echo "🤖 Installing Ollama..."
    brew install ollama
fi

# Create installation directory
INSTALL_DIR="$HOME/.shvayambhu"
mkdir -p "$INSTALL_DIR"
cd "$INSTALL_DIR"

# Clone repository
echo "📥 Downloading Shvayambhu..."
if [ -d "shvayambhu" ]; then
    cd shvayambhu
    git pull
else
    git clone https://github.com/shvayambhu/shvayambhu.git
    cd shvayambhu
fi

# Create virtual environment
echo "🔧 Setting up Python environment..."
python3 -m venv venv
source venv/bin/activate

# Install dependencies
echo "📦 Installing dependencies..."
pip install --upgrade pip
pip install mlx rich numpy

# Create launcher script
echo "🚀 Creating launcher..."
cat > "$HOME/.local/bin/shvayambhu" << 'EOF'
#!/bin/bash
cd "$HOME/.shvayambhu/shvayambhu"
source venv/bin/activate
python shvayambhu.py "$@"
EOF

chmod +x "$HOME/.local/bin/shvayambhu"

# Add to PATH if needed
if [[ ":$PATH:" != *":$HOME/.local/bin:"* ]]; then
    echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$HOME/.zshrc"
    echo -e "${YELLOW}ℹ️  Added ~/.local/bin to PATH. Run 'source ~/.zshrc' to update current session.${NC}"
fi

# Download base model
echo "📥 Downloading base model (this may take a few minutes)..."
ollama pull llama3.1:8b

# Create desktop app (optional)
echo "🖥️  Creating desktop app..."
cat > "$HOME/Desktop/Shvayambhu.command" << 'EOF'
#!/bin/bash
cd "$HOME/.shvayambhu/shvayambhu"
source venv/bin/activate
python shvayambhu.py
EOF
chmod +x "$HOME/Desktop/Shvayambhu.command"

echo ""
echo -e "${GREEN}✅ Shvayambhu installed successfully!${NC}"
echo ""
echo "To start using Shvayambhu:"
echo ""
echo "  1. Command line:"
echo "     shvayambhu"
echo ""
echo "  2. Desktop app:"
echo "     Double-click 'Shvayambhu' on your Desktop"
echo ""
echo "  3. Python:"
echo "     from shvayambhu import Shvayambhu"
echo "     model = Shvayambhu()"
echo "     print(model.generate('Hello!'))"
echo ""
echo "Run 'shvayambhu --help' for more options."
echo ""
echo "Enjoy your conscious AI assistant! 🎉"