#!/bin/bash

# Install Python requirements
echo "🐍 Installing Python requirements..."
pip install -r requirements.txt

# Start the Foundry server in the background
echo "🚀 Starting Foundry server..."
python foundry_server.py &

# Save the server's PID
SERVER_PID=$!

# Navigate to frontend directory
echo "📁 Moving to frontend directory..."
cd ai-foundry-frontend

# Install npm dependencies
echo "📦 Installing npm packages..."
npm install

# Start the frontend
echo "🌐 Starting frontend development server..."
npm run dev

# When the npm process ends, kill the Python server
kill $SERVER_PID