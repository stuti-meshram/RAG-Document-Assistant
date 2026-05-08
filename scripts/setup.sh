#!/usr/bin/env bash
# scripts/setup.sh
# One-shot local environment setup
set -euo pipefail

echo "═══════════════════════════════════════════"
echo "  RAG Document Assistant — Local Setup"
echo "═══════════════════════════════════════════"

# ── Python version check ──────────────────────────────────
python3 --version | grep -q "3\.\(10\|11\|12\)" || {
    echo "❌ Python 3.10+ required."
    exit 1
}

# ── Virtual environment ───────────────────────────────────
if [ ! -d ".venv" ]; then
    echo "▶ Creating virtual environment…"
    python3 -m venv .venv
fi

source .venv/bin/activate
echo "✅ Virtual environment activated."

# ── Install deps ─────────────────────────────────────────
echo "▶ Installing dependencies…"
pip install --upgrade pip --quiet
pip install -r requirements.txt --quiet
echo "✅ Dependencies installed."

# ── .env file ────────────────────────────────────────────
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo "⚠  Created .env from .env.example — please fill in your keys!"
else
    echo "✅ .env file already exists."
fi

# ── Data directory ────────────────────────────────────────
mkdir -p data/faiss_index
echo "✅ Data directory ready."

echo ""
echo "═══════════════════════════════════════════"
echo "  Setup complete! Next steps:"
echo "  1. Edit .env with your OpenAI API key"
echo "  2. (Optional) Add AWS credentials to .env"
echo "  3. Run:  streamlit run app/main.py"
echo "═══════════════════════════════════════════"
