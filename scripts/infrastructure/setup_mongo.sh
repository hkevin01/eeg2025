#!/bin/bash
# Simplified MongoDB-only setup for EEG2025 experiment tracking

set -e

echo "🚀 Setting up MongoDB for EEG2025 experiment tracking..."
echo ""

# Check Docker
if ! command -v docker &> /dev/null; then
    echo "❌ Docker required. Install: https://docs.docker.com/get-docker/"
    exit 1
fi

# Navigate to project root
cd "$(dirname "$0")/../.."

# Start MongoDB
echo "🐳 Starting MongoDB..."
docker-compose -f docker/docker-compose.mongo.yml up -d

echo "⏳ Waiting for MongoDB..."
sleep 5

# Test connection
if docker exec eeg2025-mongo mongosh --eval "db.adminCommand('ping')" > /dev/null 2>&1; then
    echo "✅ MongoDB is ready"
else
    echo "❌ MongoDB failed to start"
    exit 1
fi

# Install Python dependency
pip install --quiet pymongo 2>/dev/null || pip install --break-system-packages pymongo

# Create .env
if [ ! -f .env ]; then
    echo "MONGO_URI=mongodb://localhost:27017/" > .env
fi

# Test Python connection
python3 -c "from src.data.nosql_backend import MongoExperimentTracker; m = MongoExperimentTracker(); m.close(); print('✅ Python connection OK')"

echo ""
echo "🎉 Setup complete!"
echo ""
echo "📊 MongoDB:     http://localhost:27017"
echo "🌐 Web UI:      http://localhost:8082 (admin/pass123)"
echo ""
echo "🛑 Stop:        docker-compose -f docker/docker-compose.mongo.yml stop"
echo "🗑️  Remove all:  docker-compose -f docker/docker-compose.mongo.yml down -v"
echo ""
