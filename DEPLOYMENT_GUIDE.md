# Asmeranda Modular Application - Deployment Guide

## 🚀 Deployment Options

This application supports three deployment methods:

1. **Local Deployment** - Direct execution without Docker
2. **Docker Desktop** - Using Docker Compose
3. **Cloud Deployment** - AWS (Elastic Beanstalk) or GCP (Cloud Run)

---

## 📋 Prerequisites

### For All Deployments:
- Python 3.11+
- Node.js 18+
- Git

### For Docker Desktop:
- Docker Desktop installed and running

### For Cloud Deployment:
- AWS CLI (for AWS deployment)
- Google Cloud SDK (for GCP deployment)
- Cloud provider account with appropriate permissions

---

## 🖥️ Local Deployment

### Quick Start:

#### Linux/Mac:
```bash
chmod +x deploy-local.sh
./deploy-local.sh
```

#### Windows (PowerShell):
```powershell
# First, set up backend
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements-backend.txt
python -m uvicorn main:app --host 0.0.0.0 --port 8000

# In another terminal, set up frontend
cd frontend
npm install
npm run dev
```

### Access:
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API Documentation: http://localhost:8000/docs

### Stop:
Press Ctrl+C in both terminal windows

---

## 🐳 Docker Desktop Deployment

### Quick Start:

#### Linux/Mac:
```bash
chmod +x deploy-docker-desktop.sh
./deploy-docker-desktop.sh
```

#### Windows (PowerShell):
```powershell
.\deploy-docker-desktop.ps1
```

### Manual Docker Compose:
```bash
# Build and start services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down

# Rebuild after changes
docker-compose up -d --build
```

### Access:
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- Nginx (if enabled): http://localhost

### Container Management:
```bash
# View running containers
docker ps

# View logs for specific service
docker-compose logs backend
docker-compose logs frontend

# Restart specific service
docker-compose restart backend
```

---

## ☁️ Cloud Deployment

### AWS Deployment (Elastic Beanstalk)

#### Prerequisites:
```bash
# Install AWS CLI
pip install awscli

# Configure AWS credentials
aws configure
```

#### Deployment:
```bash
chmod +x deploy-cloud-aws.sh
./deploy-cloud-aws.sh
```

#### Configuration:
- Set your AWS Account ID in the script
- Choose your preferred AWS region
- Ensure you have Elastic Beanstalk permissions

#### Access:
- Application URL: Provided after deployment
- Health monitoring: AWS Elastic Beanstalk console

### GCP Deployment (Cloud Run)

#### Prerequisites:
```bash
# Install Google Cloud SDK
# https://cloud.google.com/sdk/docs/install

# Authenticate
gcloud auth login
gcloud config set project YOUR_PROJECT_ID
```

#### Deployment:
```bash
chmod +x deploy-cloud-gcp.sh
./deploy-cloud-gcp.sh
```

#### Configuration:
- Set your GCP Project ID
- Choose your preferred region
- Ensure Cloud Run and Cloud Build APIs are enabled

#### Access:
- Application URL: Provided after deployment
- Monitor: Google Cloud Console → Cloud Run

---

## 🔧 Configuration

### Environment Variables

#### Backend (`.env`):
```bash
LOG_LEVEL=INFO
DEBUG=false
DATA_DIR=./data
MAX_UPLOAD_SIZE_MB=100
CORS_ORIGINS=*
```

#### Frontend (`.env.local`):
```bash
NEXT_PUBLIC_API_BASE_PATH=http://localhost:8000/api/v1
```

### Production Configuration

#### Backend Production Settings:
```python
# backend/core/config.py
debug = False
log_level = "INFO"
cors_origins = ["https://yourdomain.com"]
```

#### Frontend Production Settings:
```bash
# frontend/.env.production
NEXT_PUBLIC_API_BASE_PATH=https://api.yourdomain.com/api/v1
```

---

## 📊 Monitoring & Logging

### Docker Desktop:
```bash
# View logs
docker-compose logs -f backend
docker-compose logs -f frontend

# Resource usage
docker stats
```

### Cloud Deployment:
- **AWS**: CloudWatch logs and metrics
- **GCP**: Cloud Logging and Monitoring

### Health Checks:
- Backend: `GET /health`
- Frontend: Check service status in cloud console

---

## 🔒 Security Considerations

### For Production:
1. Enable HTTPS (SSL/TLS)
2. Configure proper CORS origins
3. Set up authentication/authorization
4. Use environment variables for secrets
5. Enable rate limiting
6. Regular security updates

### Docker Security:
```bash
# Scan images for vulnerabilities
docker scan $APP_NAME-backend
docker scan $APP_NAME-frontend
```

---

## 🗄️ Database Storage

### Current Setup:
- Uses file-based storage in `data/` directory
- State persistence in `data/states/`
- Dataset storage in `data/datasets/`
- Model storage in `data/models/`

### Production Recommendations:
1. Use cloud storage (S3, GCS, Azure Blob)
2. Use managed databases (PostgreSQL, MongoDB)
3. Implement proper backup strategies
4. Use distributed caching (Redis)

---

## 🧪 Testing Deployments

### Local Testing:
```bash
# Test backend health
curl http://localhost:8000/health

# Test API endpoints
curl http://localhost:8000/api/v1/health

# Test frontend
open http://localhost:3000
```

### Docker Testing:
```bash
# Test container health
docker exec asmeranda-backend curl http://localhost:8000/health

# Test network connectivity
docker-compose exec backend ping frontend
```

### Cloud Testing:
- Use cloud provider's console to test
- Check health endpoints
- Monitor logs for errors
- Test all critical user flows

---

## 🚨 Troubleshooting

### Common Issues:

#### Backend won't start:
- Check port 8000 is not in use
- Verify Python dependencies are installed
- Check logs for specific errors

#### Frontend won't start:
- Check port 3000 is not in use
- Verify Node.js dependencies are installed
- Clear `.next` cache: `rm -rf .next`

#### Docker issues:
- Ensure Docker Desktop is running
- Check Docker logs: `docker-compose logs`
- Rebuild images: `docker-compose build --no-cache`

#### Cloud deployment issues:
- Verify cloud provider credentials
- Check quota limits
- Review cloud provider logs
- Ensure network/firewall settings

---

## 📈 Performance Optimization

### Backend:
- Use gunicorn for production (instead of uvicorn)
- Enable caching
- Use async operations where possible
- Optimize database queries

### Frontend:
- Use production build (`npm run build`)
- Enable static asset optimization
- Use CDN for static assets
- Implement code splitting

---

## 🔄 CI/CD Integration

### GitHub Actions Example:
```yaml
name: Deploy
on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Deploy to AWS
        run: ./deploy-cloud-aws.sh
```

---

## 📞 Support

For deployment issues:
1. Check this guide first
2. Review logs for specific errors
3. Test locally before cloud deployment
4. Ensure all prerequisites are met

---

## 🎯 Success Criteria

Deployment is successful when:
- ✅ Backend health endpoint returns 200
- ✅ Frontend loads without errors
- ✅ API endpoints are accessible
- ✅ All core features work end-to-end
- ✅ No console errors in browser
- ✅ Application responds within 2 seconds