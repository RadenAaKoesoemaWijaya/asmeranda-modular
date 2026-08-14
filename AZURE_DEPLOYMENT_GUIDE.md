# Panduan Deploy Aplikasi Asmeranda ML ke Microsoft Azure

## 📋 Daftar Isi
1. [Prasyarat](#prasyarat)
2. [Persiapan Environment](#persiapan-environment)
3. [Build Docker Image](#build-docker-image)
4. [Deploy ke Azure Container Apps](#deploy-ke-azure-container-apps)
5. [Konfigurasi Azure Services](#konfigurasi-azure-services)
6. [Monitoring dan Logging](#monitoring-dan-logging)
7. [Optimasi Performa](#optimasi-performa)
8. [Troubleshooting](#troubleshooting)
9. [Security Best Practices](#security-best-practices)

## 🎯 Prasyarat

### Tools yang Diperlukan:
- **Azure CLI** (versi terbaru)
- **Docker Desktop** (versi terbaru)
- **Git** (untuk version control)
- **Python 3.9+** (untuk testing lokal)
- **Kubectl** (opsional, untuk debugging)

### Azure Subscription:
- Azure subscription aktif
- Permissions: Contributor atau Owner pada resource group
- Azure Container Registry (ACR) quota tersedia

### Sistem Requirements:
- RAM: Minimal 8GB (disarankan 16GB)
- Storage: Minimal 10GB free space
- Internet: Koneksi stabil untuk upload Docker image

## 🔧 Persiapan Environment

### 1. Login ke Azure
```bash
# Login ke Azure
az login

# Set subscription (jika punya multiple subscriptions)
az account set --subscription "YOUR_SUBSCRIPTION_ID"

# Verifikasi login
az account show
```

### 2. Set Environment Variables
```bash
# Set environment variables
export RESOURCE_GROUP="asmeranda-ml-rg"
export LOCATION="eastus"
export ACR_NAME="asmerandacr$(date +%s)"
export CONTAINER_APP_NAME="asmeranda-ml-app"
export CONTAINER_ENV_NAME="asmeranda-container-env"
```

### 3. Create Resource Group
```bash
# Create resource group
az group create --name $RESOURCE_GROUP --location $LOCATION
```

## 🐳 Build Docker Image

### 1. Test Locally First
```bash
# Build Docker image locally
docker build -f Dockerfile.azure -t asmeranda-ml-app:latest .

# Test locally
docker run -p 8501:8501 asmeranda-ml-app:latest

# Test health endpoint
curl http://localhost:8501/_stcore/health
```

### 2. Create Azure Container Registry
```bash
# Create ACR
az acr create --resource-group $RESOURCE_GROUP --name $ACR_NAME --sku Basic --location $LOCATION

# Login ke ACR
az acr login --name $ACR_NAME

# Tag image untuk ACR
docker tag asmeranda-ml-app:latest $ACR_NAME.azurecr.io/asmeranda-ml-app:latest

# Push ke ACR
docker push $ACR_NAME.azurecr.io/asmeranda-ml-app:latest
```

## ☁️ Deploy ke Azure Container Apps

### 1. Create Container Apps Environment
```bash
# Create Container Apps Environment
az containerapp env create \
    --name $CONTAINER_ENV_NAME \
    --resource-group $RESOURCE_GROUP \
    --location $LOCATION
```

### 2. Deploy Container App
```bash
# Deploy menggunakan ARM template
az deployment group create \
    --resource-group $RESOURCE_GROUP \
    --template-file azure-infrastructure.json \
    --parameters \
        appName=$CONTAINER_APP_NAME \
        environmentName=$CONTAINER_ENV_NAME \
        acrName=$ACR_NAME \
        location=$LOCATION
```

### 3. Alternative: Deploy menggunakan Azure CLI
```bash
# Deploy langsung menggunakan CLI
az containerapp create \
    --name $CONTAINER_APP_NAME \
    --resource-group $RESOURCE_GROUP \
    --environment $CONTAINER_ENV_NAME \
    --image $ACR_NAME.azurecr.io/asmeranda-ml-app:latest \
    --target-port 8501 \
    --ingress external \
    --min-replicas 1 \
    --max-replicas 5 \
    --cpu 2.0 \
    --memory 4Gi
```

## ⚙️ Konfigurasi Azure Services

### 1. Azure Key Vault (Security)
```bash
# Create Key Vault
az keyvault create --resource-group $RESOURCE_GROUP --name asmeranda-kv --location $LOCATION

# Store secrets
az keyvault secret set --vault-name asmeranda-kv --name "super-admin-password" --value "Admin@12345"
az keyvault secret set --vault-name asmeranda-kv --name "smtp-password" --value "your-smtp-password"
az keyvault secret set --vault-name asmeranda-kv --name "azure-storage-connection" --value "your-storage-connection-string"
```

### 2. Azure Storage Account (File Storage)
```bash
# Create storage account
az storage account create \
    --resource-group $RESOURCE_GROUP \
    --name asmerandasa$(date +%s) \
    --location $LOCATION \
    --sku Standard_LRS \
    --kind StorageV2

# Create blob container
az storage container create \
    --name models \
    --account-name your-storage-account
```

### 3. Azure Database for PostgreSQL (Optional)
```bash
# Create PostgreSQL server (jika ingin migrate dari SQLite)
az postgres server create \
    --resource-group $RESOURCE_GROUP \
    --name asmeranda-postgres \
    --location $LOCATION \
    --admin-user asmerandaadmin \
    --admin-password YourSecurePassword123! \
    --sku-name B_Gen5_2 \
    --version 13
```

## 📊 Monitoring dan Logging

### 1. Application Insights
```bash
# Create Application Insights
az monitor app-insights component create \
    --app asmeranda-app-insights \
    --location $LOCATION \
    --resource-group $RESOURCE_GROUP

# Get instrumentation key
APP_INSIGHTS_KEY=$(az monitor app-insights component show \
    --app asmeranda-app-insights \
    --resource-group $RESOURCE_GROUP \
    --query instrumentationKey -o tsv)
```

### 2. Log Analytics Workspace
```bash
# Create Log Analytics Workspace
az monitor log-analytics workspace create \
    --resource-group $RESOURCE_GROUP \
    --workspace-name asmeranda-law \
    --location $LOCATION
```

### 3. Container App Logs
```bash
# Stream logs from container app
az containerapp logs show \
    --name $CONTAINER_APP_NAME \
    --resource-group $RESOURCE_GROUP \
    --follow
```

## 🚀 Optimasi Performa

### 1. Scaling Configuration
```yaml
# Update container app scaling
az containerapp update \
    --name $CONTAINER_APP_NAME \
    --resource-group $RESOURCE_GROUP \
    --min-replicas 2 \
    --max-replicas 10 \
    --scale-rule-name http-rule \
    --scale-rule-type http \
    --scale-rule-metadata concurrentRequests=10
```

### 2. Resource Optimization
```yaml
# Update CPU and memory
az containerapp update \
    --name $CONTAINER_APP_NAME \
    --resource-group $RESOURCE_GROUP \
    --cpu 4.0 \
    --memory 8Gi
```

### 3. Environment Variables for Performance
```yaml
# Set performance-related environment variables
az containerapp update \
    --name $CONTAINER_APP_NAME \
    --resource-group $RESOURCE_GROUP \
    --set-env-vars \
        STREAMLIT_SERVER_MAX_UPLOAD_SIZE=500 \
        STREAMLIT_SERVER_ENABLE_CORS=false \
        STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=true
```

## 🔧 Troubleshooting

### Common Issues:

#### 1. Container Startup Issues
```bash
# Check container logs
az containerapp logs show \
    --name $CONTAINER_APP_NAME \
    --resource-group $RESOURCE_GROUP

# Check container status
az containerapp show \
    --name $CONTAINER_APP_NAME \
    --resource-group $RESOURCE_GROUP \
    --query properties.provisioningState
```

#### 2. Health Check Failures
```bash
# Test health endpoint manually
APP_URL=$(az containerapp show \
    --name $CONTAINER_APP_NAME \
    --resource-group $RESOURCE_GROUP \
    --query properties.configuration.ingress.fqdn -o tsv)

curl https://$APP_URL/_stcore/health
```

#### 3. Memory Issues
```bash
# Monitor resource usage
az containerapp revision list \
    --name $CONTAINER_APP_NAME \
    --resource-group $RESOURCE_GROUP \
    --query "[].properties.resourceConsumption"
```

#### 4. ACR Authentication Issues
```bash
# Re-authenticate to ACR
az acr login --name $ACR_NAME

# Check ACR credentials
az acr credential show --name $ACR_NAME
```

## 🔒 Security Best Practices

### 1. Network Security
```bash
# Enable HTTPS only
az containerapp update \
    --name $CONTAINER_APP_NAME \
    --resource-group $RESOURCE_GROUP \
    --ingress external \
    --transport auto

# Configure CORS properly
az containerapp update \
    --name $CONTAINER_APP_NAME \
    --resource-group $RESOURCE_GROUP \
    --set-env-vars STREAMLIT_SERVER_ENABLE_CORS=false
```

### 2. Secret Management
```bash
# Use Key Vault references
az containerapp update \
    --name $CONTAINER_APP_NAME \
    --resource-group $RESOURCE_GROUP \
    --secrets \
        smtp-password=keyvaultref:https://asmeranda-kv.vault.azure.net/secrets/smtp-password \
    --set-env-vars SMTP_PASSWORD=secretref:smtp-password
```

### 3. Container Security
```dockerfile
# Use non-root user (already implemented in Dockerfile)
USER appuser

# Scan image for vulnerabilities
az acr task create \
    --registry $ACR_NAME \
    --name security-scan \
    --image asmeranda-ml-app:latest \
    --context https://github.com/your-repo.git \
    --file Dockerfile.azure \
    --git-access-token your-github-token
```

## 📋 Deployment Checklist

### Pre-Deployment:
- [ ] Semua dependencies ter-install
- [ ] Docker image berhasil di-build dan di-test lokal
- [ ] Environment variables sudah di-set
- [ ] Azure CLI sudah login

### During Deployment:
- [ ] Resource group sudah dibuat
- [ ] ACR sudah dibuat dan image sudah di-push
- [ ] Container Apps Environment sudah dibuat
- [ ] Container App deployment berhasil
- [ ] Health check berjalan normal

### Post-Deployment:
- [ ] Application dapat diakses via browser
- [ ] Semua fitur berfungsi dengan baik
- [ ] Monitoring dan logging aktif
- [ ] Security configuration sudah di-set
- [ ] Backup strategy sudah di-implementasi

## 🎯 Quick Deployment (One-Command)

Gunakan script otomatis untuk deployment cepat:

### Linux/Mac:
```bash
chmod +x deploy-to-azure.sh
./deploy-to-azure.sh
```

### Windows:
```cmd
deploy-to-azure.bat
```

## 📞 Support dan Kontak

Jika mengalami masalah:
1. Check logs menggunakan `az containerapp logs`
2. Review ARM template di `azure-infrastructure.json`
3. Test lokal menggunakan `docker-compose.azure.yml`
4. Hubungi development team untuk bantuan

---

**Catatan**: Deployment ke Azure memerlukan biaya. Pastikan untuk:
- Monitor usage dan cost
- Set budget alerts
- Cleanup resources yang tidak digunakan
- Gunakan Azure Cost Management untuk monitoring