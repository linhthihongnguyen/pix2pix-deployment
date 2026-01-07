# Setup Guide - 15 Minutes

## What Changed

### Updated Your Files:
1. ✅ **requirements.txt** - Added AWS packages (boto3, python-dotenv)
2. ✅ **app.py** - Added `.env` support (minimal changes)
3. ✅ **test_generate.py** - Auto-detects port 8000 or 5000

### New Files (Only 3!):
4. ✅ **README.md** - Project documentation
5. ✅ **.gitignore** - Prevents committing model/secrets
6. ✅ **.env.example** - Template for credentials

## Step-by-Step Setup

### 1. Install New Dependencies (2 min)

```bash
pip install boto3 botocore python-dotenv requests
```

Or reinstall everything:
```bash
pip install -r requirements.txt
```

### 2. Create .env File (2 min)

```bash
# Copy template
cp .env.example .env

# Edit with your credentials
nano .env  # or use any text editor
```

Add your actual AWS credentials:
```env
AWS_REGION=us-east-1
AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE
AWS_SECRET_ACCESS_KEY=wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY
PORT=8000
MODEL_PATH=final_model.pth
```

### 3. Test Everything (5 min)

```bash
# Start server
python app.py

# In another terminal, test it
python test_generate.py

# Or manually
curl http://localhost:8000/health
```

### 4. Setup Git (5 min)

```bash
# Initialize git
git init

# Add all files
git add .

# IMPORTANT: Verify these are NOT being added
git status | grep -E "(\.pth|\.env)"
# Should show nothing

# Make first commit
git commit -m "Initial commit: Pix2Pix + AWS Rekognition"

# Create GitHub repo and push
git remote add origin https://github.com/yourusername/pix2pix-rekognition.git
git branch -M main
git push -u origin main
```

### 5. Add GitHub Description (1 min)

On GitHub:
- Description: `Pix2Pix GAN for satellite-to-map translation with AWS Rekognition`
- Topics: `machine-learning`, `computer-vision`, `pix2pix`, `aws`, `gan`

## That's It! ✅

Your project is now on GitHub with:
- ✅ Professional README
- ✅ Secure credentials (not in git)
- ✅ Model file not in git
- ✅ Working API
- ✅ Complete training notebook

## Troubleshooting

### `.env` or `final_model.pth` showing in git?

```bash
git rm --cached .env final_model.pth
git add .gitignore
git commit -m "Fix: ensure sensitive files are ignored"
```

### Port 8000 already in use?

```bash
# Kill process on port 8000
lsof -ti:8000 | xargs kill -9

# Or use different port
PORT=8001 python app.py
```

### AWS credentials not working?

Check your `.env` file:
```bash
cat .env  # Verify format is correct
```

Test AWS connection:
```bash
python -c "import boto3; boto3.client('rekognition', region_name='us-east-1').describe_projects()"
```

## What You Have Now

```
Your Project/
├── app.py ✅                          # Updated with .env support
├── requirements.txt ✅                 # Updated with AWS packages  
├── gunicorn_config.py ✅              # Your original file
├── test_generate.py ✅                # Updated to auto-detect port
├── Pix2Pix_Satellite_to_Map.ipynb ✅ # Your training notebook
├── final_model.pth ✅                 # Your trained model (not in git)
├── .env ✅                            # NEW - Your credentials (not in git)
├── .env.example ✅                    # NEW - Template (in git)
├── .gitignore ✅                      # NEW - Protects sensitive files
└── README.md ✅                       # NEW - Documentation
```

## Next Steps (Optional)

Want to add more? You can add later:
- Docker support
- More tests
- Deployment guides
- Contributing guidelines
