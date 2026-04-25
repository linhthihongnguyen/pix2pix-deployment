# AWS EC2 Deployment Guide

Complete guide for deploying the Pix2Pix + Rekognition service to AWS EC2.

**Estimated Time:** 2 hours  
**Cost:** ~$17/month (t3.small instance)  
**Prerequisites:** AWS account, basic Linux knowledge

---

## Prerequisites

- AWS account with EC2 access
- Basic familiarity with SSH and Linux commands
- Credit card for AWS billing (free tier available)

---

## Step 1: Launch EC2 Instance

### Create Instance

1. **Login to AWS Console:** https://console.aws.amazon.com
2. **Navigate to EC2:** Services → Compute → EC2
3. **Launch Instance:**
   - **Name:** `pix2pix-server`
   - **OS:** Ubuntu Server 24.04 LTS (free tier eligible)
   - **Instance Type:** t3.small (2 vCPUs, 2GB RAM)
     - ⚠️ t3.micro insufficient (1GB RAM causes OOM errors)
   - **Key Pair:** Create new → Name: `pix2pix-key` → Download .pem file
   - **Storage:** 20 GB gp3 SSD (default)

### Configure Security Group

**Add these inbound rules:**

| Type | Port | Source | Description |
|------|------|--------|-------------|
| SSH | 22 | 0.0.0.0/0 | SSH access |
| Custom TCP | 8000 | 0.0.0.0/0 | Flask application |

4. **Launch Instance**
5. **Note the Public IPv4 address** (e.g., 3.235.252.100)

---

## Step 2: Connect to Instance

### Fix Key Permissions (Windows)

```powershell
icacls pix2pix-key.pem /inheritance:r
icacls pix2pix-key.pem /grant:r "$env:USERNAME:(R)"
```

### SSH into Instance

```bash
ssh -i pix2pix-key.pem ubuntu@YOUR-PUBLIC-IP
```

---

## Step 3: Install System Dependencies

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Python and dependencies
sudo apt install python3-pip python3-venv libgl1 libglib2.0-0 screen -y
```

**Note:** Ubuntu 24.04 uses `libgl1` (not `libgl1-mesa-glx`)

---

## Step 4: Setup Application

### Create Project Directory

```bash
mkdir ~/pix2pix-app
cd ~/pix2pix-app
```

### Transfer Code

**From your local computer:**

```bash
# Transfer application code
scp -i pix2pix-key.pem app.py ubuntu@YOUR-IP:~/pix2pix-app/
scp -i pix2pix-key.pem requirements.txt ubuntu@YOUR-IP:~/pix2pix-app/
scp -i pix2pix-key.pem gunicorn_config.py ubuntu@YOUR-IP:~/pix2pix-app/

# Transfer model file (219 MB - this will take a few minutes)
scp -i pix2pix-key.pem final_model.pth ubuntu@YOUR-IP:~/pix2pix-app/
```

**Note:** Model file transfer takes 5-10 minutes depending on connection speed.

---

## Step 5: Setup Python Environment

**On EC2 instance:**

```bash
cd ~/pix2pix-app

# Create virtual environment
python3 -m venv venv

# Activate
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt
```

**Installation takes ~10 minutes** (PyTorch is large)

---

## Step 6: Configure AWS Rekognition

### Create IAM Role

1. **Go to IAM Console:** Services → Security → IAM
2. **Create Role:**
   - Trusted entity: AWS service → EC2
   - Permissions: `AmazonRekognitionFullAccess`
   - Name: `pix2pix-rekognition-role`
3. **Attach to EC2:**
   - EC2 → Instances → Select your instance
   - Actions → Security → Modify IAM role
   - Select: `pix2pix-rekognition-role`

**This allows your app to access Rekognition without hardcoded credentials.** ✅

---

## Step 7: Start Application

### Using Screen (Persistent Session)

```bash
# Create screen session
screen -S pix2pix

# Activate venv
cd ~/pix2pix-app
source venv/bin/activate

# Start Gunicorn
gunicorn app:app --bind 0.0.0.0:8000 --timeout 300 --workers 1 --preload
```

**Wait for:**
[INFO] Listening at: http://0.0.0.0:8000
✓ Pix2Pix model loaded successfully
✓ AWS Rekognition client initialized successfully

**Detach from screen:** Press `Ctrl+A`, then `D`

**Exit SSH:** Type `exit`

---

## Step 8: Test Deployment

### From Your Local Computer

```bash
# Health check
curl http://YOUR-PUBLIC-IP:8000/health

# Expected response:
# {"status":"healthy","model_loaded":true,"rekognition_enabled":true}

# Service info
curl http://YOUR-PUBLIC-IP:8000/

# Generate map
curl -X POST -F "image=@satellite.jpg" \
     http://YOUR-PUBLIC-IP:8000/generate-enhanced \
     -o result.json

# Decode result
python decode.py
```

---

## Troubleshooting

### Connection Refused

**Problem:** Application not running  
**Solution:**
```bash
ssh -i pix2pix-key.pem ubuntu@YOUR-IP
screen -r pix2pix  # Reattach to check status
```

### Out of Memory Errors

**Problem:** t3.micro insufficient  
**Solution:** Upgrade to t3.small (2GB RAM required)

### Port Already in Use

**Problem:** Previous process still running  
**Solution:**
```bash
pkill -f gunicorn
sudo lsof -i :8000  # Verify port is free
```

### Rekognition Not Working

**Problem:** IAM role not attached  
**Solution:** Verify IAM role in EC2 console → Instance details

---

## Managing the Service

### Check if Running

```bash
ssh -i pix2pix-key.pem ubuntu@YOUR-IP
screen -ls

# Should show: XXXX.pix2pix (Detached)
```

### View Logs

```bash
screen -r pix2pix  # Reattach to see output
# Detach: Ctrl+A, D
```

### Restart Service

```bash
screen -X -S pix2pix quit  # Stop
screen -S pix2pix          # Start new session
# ... run gunicorn command ...
```

### Stop Service

```bash
screen -X -S pix2pix quit
```

---

## Cost Management

### Monthly Costs (t3.small, on-demand)

| Item | Cost |
|------|------|
| EC2 t3.small | ~$17/month |
| EBS Storage (20GB) | ~$2/month |
| Data Transfer (minimal) | ~$0/month |
| Rekognition | $1 per 1,000 images |

**Total:** ~$19/month base + usage

### Stopping Instance

**Stop (preserves instance):**
- AWS Console → EC2 → Instances → Instance State → Stop
- Cost after stopping: ~$2/month (storage only)

**Terminate (deletes everything):**
- AWS Console → EC2 → Instances → Instance State → Terminate
- Cost after terminating: $0/month

---

## Security Best Practices

1. ✅ Use IAM roles instead of access keys
2. ✅ Keep SSH keys secure (never commit to Git)
3. ✅ Restrict security group to specific IPs (optional)
4. ✅ Regular system updates: `sudo apt update && sudo apt upgrade`
5. ✅ Monitor billing: Set up AWS billing alerts

---

## Architecture
User Request
↓
Internet
↓
AWS EC2 t3.small
↓
Flask + Gunicorn (Port 8000)
↓
┌─────────────────────────────────┐
│  Pix2Pix GAN (PyTorch)          │
│  + AWS Rekognition API          │
└─────────────────────────────────┘
↓
Response (Generated Map)

---

## Performance Expectations

- **First request:** 3-5 seconds (cold start)
- **Subsequent requests:** 2-3 seconds average
- **Memory usage:** ~1.2 GB during inference
- **CPU usage:** ~80% during processing

---

## Next Steps

After successful deployment:

1. ✅ Test all endpoints
2. ✅ Set up CloudWatch monitoring (optional)
3. ✅ Configure automated backups (optional)
4. ✅ Document your public IP for API consumers

---

## Author

Nguyen Thi Hong Linh
**GitHub:** [@linhthihongnguyen](https://github.com/linhthihongnguyen)