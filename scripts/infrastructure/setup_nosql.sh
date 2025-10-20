#!/bin/bash
# Quick setup script for NoSQL integration

set -e

echo "================================================================================"
echo "🚀 EEG2025 NoSQL Integration Setup"
echo "================================================================================"
echo ""

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

echo "✅ Docker found"
echo ""

# Check if docker-compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "❌ docker-compose is not installed. Please install docker-compose first."
    exit 1
fi

echo "✅ docker-compose found"
echo ""

# Navigate to project root
cd "$(dirname "$0")/../.."

# Start NoSQL services
echo "🐳 Starting MongoDB and Redis containers..."
cd docker
docker-compose -f docker-compose.nosql.yml up -d

echo ""
echo "⏳ Waiting for services to be ready..."
sleep 10

# Check MongoDB
echo "🔍 Checking MongoDB..."
if docker exec eeg2025-mongo mongosh --eval "db.adminCommand('ping')" > /dev/null 2>&1; then
    echo "✅ MongoDB is running"
else
    echo "❌ MongoDB is not responding"
    exit 1
fi

# Check Redis
echo "🔍 Checking Redis..."
if docker exec eeg2025-redis redis-cli ping > /dev/null 2>&1; then
    echo "✅ Redis is running"
else
    echo "❌ Redis is not responding"
    exit 1
fi

cd ..

# Install Python dependencies
echo ""
echo "📦 Installing Python dependencies..."
pip install --quiet pymongo redis || pip install --break-system-packages pymongo redis

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo ""
    echo "📝 Creating .env file..."
    cat > .env << 'ENVEOF'
# NoSQL Configuration
MONGO_URI=mongodb://localhost:27017/
REDIS_HOST=localhost
REDIS_PORT=6379
ENVEOF
    echo "✅ Created .env file"
fi

# Test connections
echo ""
echo "🧪 Testing connections..."
python3 << 'PYEOF'
import sys
try:
    from src.data.nosql_backend import MongoExperimentTracker
    from src.data.redis_cache import RedisFeatureCache
    
    print("Testing MongoDB connection...")
    mongo = MongoExperimentTracker()
    print("✅ MongoDB connection successful")
    mongo.close()
    
    print("Testing Redis connection...")
    redis_cache = RedisFeatureCache()
    print("✅ Redis connection successful")
    redis_cache.close()
    
    print("\n🎉 All connections successful!")
except Exception as e:
    print(f"\n❌ Connection test failed: {e}")
    sys.exit(1)
PYEOF

echo ""
echo "================================================================================"
echo "✅ NoSQL Integration Setup Complete!"
echo "================================================================================"
echo ""
echo "📊 Services Running:"
echo "   • MongoDB:         http://localhost:27017"
echo "   • Redis:           http://localhost:6379"
echo "   • MongoDB UI:      http://localhost:8082 (admin/pass123)"
echo "   • Redis Commander: http://localhost:8081"
echo ""
echo "🛠️  Management Commands:"
echo "   • Stop services:   docker-compose -f docker/docker-compose.nosql.yml stop"
echo "   • Start services:  docker-compose -f docker/docker-compose.nosql.yml start"
echo "   • View logs:       docker-compose -f docker/docker-compose.nosql.yml logs -f"
echo "   • Remove all:      docker-compose -f docker/docker-compose.nosql.yml down -v"
echo ""
echo "📚 Next Steps:"
echo "   1. Read docs/NOSQL_ML_INTEGRATION.md for architecture details"
echo "   2. Use MongoExperimentTracker in your training scripts"
echo "   3. Check the web UIs to explore your data"
echo ""
