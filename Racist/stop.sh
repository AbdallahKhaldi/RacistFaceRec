#!/bin/bash

# Stop the Facial Recognition Server

echo "Stopping Facial Recognition Server..."

# Find and kill the process on port 5001
PID=$(lsof -ti:5001)

if [ -z "$PID" ]; then
    echo "No server running on port 5001"
else
    kill -9 $PID
    echo "Server stopped (PID: $PID)"
fi
