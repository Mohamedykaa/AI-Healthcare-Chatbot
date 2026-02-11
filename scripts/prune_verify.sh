#!/bin/bash
# Verification script for post-cleanup validation
# Run: bash scripts/prune_verify.sh

echo "=== Project Cleanup Verification ==="

echo ""
echo "1. Checking backend imports..."
python -c "from backend.app import app; print('✅ Backend imports OK')"

echo ""
echo "2. Checking frontend imports..."
python -c "import streamlit; print('✅ Streamlit available')"

echo ""
echo "3. Running pytest on core tests..."
pytest tests/test_backend_flow.py tests/test_menu_parsing.py -v --tb=short

echo ""
echo "4. Verifying archive structure..."
if [ -d "archive/legacy" ]; then
    echo "✅ archive/legacy exists"
else
    echo "❌ archive/legacy missing"
fi

if [ -d "archive/legacy_src" ]; then
    echo "✅ archive/legacy_src exists"
else
    echo "❌ archive/legacy_src missing"
fi

echo ""
echo "=== Verification Complete ==="
