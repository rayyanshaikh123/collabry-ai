# Collabry AI Engine - Production Deployment Guide

This guide covers best practices for deploying the Collabry AI Engine to production.

## 📋 Pre-Deployment Checklist

### Security
- [ ] Change `JWT_SECRET` to a cryptographically secure random string (32+ characters)
- [ ] Enable HTTPS/SSL for all endpoints
- [ ] Configure firewall rules for MongoDB and Redis
- [ ] Set up MongoDB authentication with strong passwords
- [ ] Enable Redis password authentication
- [ ] Review and minimize CORS origins list
- [ ] Set appropriate file upload size limits
- [ ] Enable rate limiting with appropriate thresholds
- [ ] Implement request logging and monitoring

### Configuration
- [ ] Set `LOG_LEVEL=INFO` or `WARNING` (not `DEBUG`)
- [ ] Configure production database URLs (MongoDB Atlas recommended)
- [ ] Set up Redis cache (Redis Cloud or Upstash recommended)
- [ ] Configure LLM API keys with production quotas
- [ ] Set appropriate worker count for your server size
- [ ] Configure automatic backups for MongoDB
- [ ] Set up health check monitoring
- [ ] Configure error alerting (e.g., Sentry)

### Performance
- [ ] Enable Redis caching for frequently accessed data
- [ ] Configure appropriate connection pooling for MongoDB
- [ ] Set up CDN for static assets (if any)
- [ ] Optimize vector database for production workloads
- [ ] Configure appropriate timeouts for long-running operations
- [ ] Set up horizontal scaling if needed

---

## 🐳 Docker Deployment (Recommended)

### Using Docker Compose

1. **Update Environment Variables**
   ```bash
   cp .env.example .env
   # Edit .env with production values
   ```

2. **Build and Start Services**
   ```bash
   docker-compose up -d --build
   ```

3. **Verify Deployment**
   ```bash
   # Check service status
   docker-compose ps
   
   # Check logs
   docker-compose logs -f ai-engine
   
   # Test health endpoint
   curl http://localhost:8000/health
   ```

4. **Update Services**
   ```bash
   # Pull latest code
   git pull origin main
   
   # Rebuild and restart
   docker-compose up -d --build ai-engine
   ```

### Production Docker Configuration

**docker-compose.prod.yml** (create this file):
```yaml
version: '3.8'

services:
  ai-engine:
    build:
      context: ./ai-engine
      dockerfile: Dockerfile
    container_name: collabry-ai-engine
    restart: unless-stopped
    ports:
      - "8000:8000"
    environment:
      - MONGO_URI=${MONGO_URI}
      - REDIS_URL=${REDIS_URL}
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - JWT_SECRET=${JWT_SECRET}
      - LOG_LEVEL=INFO
      - WORKERS=4
    volumes:
      - ./ai-engine/documents:/app/documents
      - ./ai-engine/memory:/app/memory
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
    networks:
      - backend
    depends_on:
      - redis
      - mongodb

  redis:
    image: redis:7-alpine
    container_name: collabry-redis
    restart: unless-stopped
    command: redis-server --requirepass ${REDIS_PASSWORD}
    volumes:
      - redis-data:/data
    networks:
      - backend

  mongodb:
    image: mongo:7
    container_name: collabry-mongodb
    restart: unless-stopped
    environment:
      MONGO_INITDB_ROOT_USERNAME: ${MONGO_USER}
      MONGO_INITDB_ROOT_PASSWORD: ${MONGO_PASSWORD}
    volumes:
      - mongo-data:/data/db
    networks:
      - backend

volumes:
  redis-data:
  mongo-data:

networks:
  backend:
    driver: bridge
```

**Usage:**
```bash
docker-compose -f docker-compose.prod.yml up -d
```

---

## ☁️ Cloud Platform Deployment

### Render.com

1. **Create Web Service**
   - Go to Render Dashboard
   - Click "New" → "Web Service"
   - Connect GitHub repository
   - Select `ai-engine` directory

2. **Configure Build Settings**
   ```
   Build Command: pip install -r requirements.txt
   Start Command: gunicorn server.main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:$PORT
   ```

3. **Environment Variables**
   Add all variables from `.env.example`:
   ```
   MONGO_URI=mongodb+srv://...
   REDIS_URL=redis://...
   OPENAI_API_KEY=sk-...
   JWT_SECRET=your-secret
   LLM_MODEL=gpt-4o-mini
   CORS_ORIGINS=https://your-domain.com
   LOG_LEVEL=INFO
   RATE_LIMIT_ENABLED=true
   ```

4. **Add Redis**
   - Go to "Add Service" → "Redis"
   - Copy the internal connection string
   - Add to `REDIS_URL` environment variable

5. **Deploy**
   - Render automatically deploys on push
   - Monitor deployment in dashboard
   - Check logs for any issues

### Railway.app

1. **Create New Project**
   - Click "New Project"
   - Select "Deploy from GitHub"
   - Choose your repository

2. **Configure Service**
   - Railway auto-detects Dockerfile
   - Set root directory to `ai-engine`
   - Add environment variables

3. **Add Database Services**
   ```bash
   # Add Redis
   railway add redis
   
   # Add MongoDB (use plugin or external)
   railway add mongodb
   ```

4. **Environment Variables**
   - Railway automatically injects `PORT`
   - Add your custom variables
   - Use Railway's variable referencing: `${{Redis.REDIS_URL}}`

5. **Deploy**
   ```bash
   # Deploy manually
   railway up
   
   # Or enable auto-deploy on push
   ```

### DigitalOcean App Platform

1. **Create App**
   - Go to Apps → Create App
   - Connect GitHub repository
   - Select `ai-engine` directory

2. **Configure Component**
   ```yaml
   name: ai-engine
   type: web
   run_command: gunicorn server.main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8080
   build_command: pip install -r requirements.txt
   http_port: 8080
   instance_count: 2
   instance_size: professional-xs
   ```

3. **Add Managed Databases**
   - MongoDB: Add managed MongoDB cluster
   - Redis: Add managed Redis cluster
   - Connection strings auto-injected

4. **Environment Variables**
   - Add all required variables
   - Use database connection strings from DigitalOcean

5. **Deploy**
   - App Platform auto-deploys on push
   - Scale instances as needed

### Heroku

1. **Login and Create App**
   ```bash
   heroku login
   heroku create collabry-ai-engine
   ```

2. **Add Addons**
   ```bash
   # MongoDB
   heroku addons:create mongolab:sandbox
   
   # Redis
   heroku addons:create heroku-redis:mini
   ```

3. **Set Environment Variables**
   ```bash
   heroku config:set OPENAI_API_KEY=your-key
   heroku config:set JWT_SECRET=your-secret
   heroku config:set LLM_MODEL=gpt-4o-mini
   heroku config:set WORKERS=4
   ```

4. **Create Procfile** (if not exists)
   ```
   web: gunicorn server.main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:$PORT
   ```

5. **Deploy**
   ```bash
   git subtree push --prefix ai-engine heroku main
   ```

---

## 🔒 Production Security Hardening

### 1. Environment Variables

Never hardcode secrets:
```python
# ❌ Bad
JWT_SECRET = "my-secret-123"

# ✅ Good
JWT_SECRET = os.getenv("JWT_SECRET")
if not JWT_SECRET:
    raise ValueError("JWT_SECRET environment variable not set")
```

### 2. Rate Limiting

Configure aggressive rate limiting:
```env
RATE_LIMIT_ENABLED=true
RATE_LIMIT_PER_MINUTE=60
RATE_LIMIT_PER_HOUR=1000
RATE_LIMIT_PER_DAY=10000
```

### 3. CORS Configuration

Whitelist only your domains:
```env
# ❌ Bad (allows all origins)
CORS_ORIGINS=*

# ✅ Good (specific domains only)
CORS_ORIGINS=https://app.yourdomain.com,https://www.yourdomain.com
```

### 4. MongoDB Security

Enable authentication:
```bash
# MongoDB Atlas: Enable IP whitelist
# Local MongoDB: Create admin user
mongosh admin --eval "db.createUser({
  user: 'admin',
  pwd: 'SecurePassword123!',
  roles: ['root']
})"
```

### 5. Redis Security

Enable password authentication:
```bash
# redis.conf
requirepass YourSecureRedisPassword

# Connection string
REDIS_URL=redis://:YourSecureRedisPassword@host:6379/0
```

### 6. File Upload Security

Configure size limits:
```python
# In server middleware
MAX_UPLOAD_SIZE = 10 * 1024 * 1024  # 10MB
ALLOWED_EXTENSIONS = {'.pdf', '.txt', '.docx', '.md'}
```

---

## 📊 Monitoring & Observability

### Health Checks

Configure external monitoring:
```bash
# Uptime monitoring services
curl https://uptimerobot.com
curl https://pingdom.com

# Simple health check script
*/5 * * * * curl -f http://your-domain.com/health || echo "AI Engine down"
```

### Logging

Use structured logging in production:
```python
import logging
import json

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Log to JSON for parsing
logger.info(json.dumps({
    "event": "request",
    "user_id": user_id,
    "endpoint": "/ai/chat",
    "duration_ms": duration
}))
```

### Error Tracking

Integrate Sentry:
```python
import sentry_sdk
from sentry_sdk.integrations.fastapi import FastApiIntegration

sentry_sdk.init(
    dsn="https://your-sentry-dsn",
    integrations=[FastApiIntegration()],
    environment="production",
    traces_sample_rate=0.1
)
```

### Metrics

Track key metrics:
- Request rate (requests/second)
- Response time (p50, p90, p99)
- Error rate (errors/total requests)
- LLM token usage
- Active sessions count
- RAG retrieval latency

---

## 🔄 Backup & Recovery

### MongoDB Backups

**Automated Backups (MongoDB Atlas):**
```bash
# Enable continuous backups in Atlas dashboard
# Set retention period: 7-30 days
```

**Manual Backup:**
```bash
# Export all collections
mongodump --uri="mongodb://user:pass@host/collabry" --out=/backups/$(date +%Y%m%d)

# Restore from backup
mongorestore --uri="mongodb://user:pass@host/collabry" /backups/20260213
```

### Document Storage Backups

Backup user documents and vector indexes:
```bash
# Create backup script
#!/bin/bash
BACKUP_DIR="/backups/$(date +%Y%m%d)"
mkdir -p $BACKUP_DIR

# Backup documents
tar -czf $BACKUP_DIR/documents.tar.gz /app/documents

# Backup FAISS indexes
tar -czf $BACKUP_DIR/memory.tar.gz /app/memory

# Upload to S3/cloud storage
aws s3 cp $BACKUP_DIR s3://your-bucket/backups/$(date +%Y%m%d) --recursive
```

### Disaster Recovery

1. **Document the Recovery Process**
2. **Test Recovery Regularly**
3. **Keep Encrypted Backups Offsite**
4. **Maintain Recovery Time Objective (RTO) < 1 hour**
5. **Maintain Recovery Point Objective (RPO) < 24 hours**

---

## 📈 Performance Optimization

### 1. Connection Pooling

Configure MongoDB connection pool:
```python
from pymongo import MongoClient

client = MongoClient(
    mongo_uri,
    maxPoolSize=50,
    minPoolSize=10,
    maxIdleTimeMS=30000
)
```

### 2. Redis Caching

Cache expensive operations:
```python
import redis
import json

redis_client = redis.from_url(REDIS_URL)

def get_cached_or_compute(key, compute_fn, ttl=3600):
    cached = redis_client.get(key)
    if cached:
        return json.loads(cached)
    
    result = compute_fn()
    redis_client.setex(key, ttl, json.dumps(result))
    return result
```

### 3. Async Operations

Use async for I/O-bound operations:
```python
from fastapi import BackgroundTasks

@app.post("/ai/upload")
async def upload_document(
    file: UploadFile,
    background_tasks: BackgroundTasks
):
    # Save file immediately
    file_path = await save_file(file)
    
    # Process in background
    background_tasks.add_task(process_document, file_path)
    
    return {"message": "Upload successful", "file_id": file_id}
```

### 4. Horizontal Scaling

Deploy multiple instances:
```bash
# Docker Swarm
docker service create --replicas 4 collabry-ai-engine

# Kubernetes
kubectl scale deployment ai-engine --replicas=4
```

---

## 🧪 Testing in Production

### Smoke Tests

Run after each deployment:
```bash
#!/bin/bash
API_URL="https://api.yourdomain.com"

# Test health endpoint
curl -f $API_URL/health || exit 1

# Test authentication
TOKEN=$(curl -s -X POST $API_URL/auth/login -d '{"email":"test@example.com","password":"test"}' | jq -r .token)

# Test chat endpoint
curl -f -X POST $API_URL/ai/chat \
  -H "Authorization: Bearer $TOKEN" \
  -d '{"message":"Hello","session_id":"test"}' || exit 1

echo "✅ All smoke tests passed"
```

### Load Testing

Use tools like k6 or Locust:
```javascript
// k6 load test
import http from 'k6/http';
import { check } from 'k6';

export default function () {
  const res = http.post('https://api.yourdomain.com/ai/chat', {
    message: 'Test message',
    session_id: 'test'
  }, {
    headers: { 'Authorization': 'Bearer TOKEN' }
  });
  
  check(res, {
    'status is 200': (r) => r.status === 200,
    'response time < 2s': (r) => r.timings.duration < 2000,
  });
}
```

---

## 📞 Support & Troubleshooting

### Common Production Issues

**Issue**: High response times
- Check MongoDB connection pool exhaustion
- Monitor Redis hit rate
- Check LLM API latency
- Enable caching for frequent queries

**Issue**: Out of memory errors
- Reduce worker count
- Increase instance size
- Implement request queuing
- Add memory limits in Docker

**Issue**: Rate limit exceeded
- Increase Redis limits
- Implement token bucket algorithm
- Add user-based quotas
- Scale horizontally

**Issue**: Database connection failures
- Check MongoDB Atlas whitelist
- Verify connection string
- Monitor connection pool metrics
- Implement retry logic with backoff

---

## 📚 Additional Resources

- [Docker Best Practices](https://docs.docker.com/develop/dev-best-practices/)
- [FastAPI Deployment](https://fastapi.tiangolo.com/deployment/)
- [MongoDB Atlas Documentation](https://docs.atlas.mongodb.com/)
- [Redis Deployment Guide](https://redis.io/docs/management/)
- [12-Factor App Methodology](https://12factor.net/)

---

**Last Updated**: February 2026
**Maintained by**: Collabry Team
