# Pix2Pix Satellite-to-Map Translation with AWS Rekognition

Satellite-to-map image translation using Pix2Pix GAN with AWS Rekognition for intelligent preprocessing and terrain classification.

## Features

- **Satellite-to-Map Translation**: U-Net generator with 54.4M parameters
- **AWS Rekognition Integration**: Automated terrain classification and quality filtering
- **Dual API Endpoints**: Original endpoint + enhanced with Rekognition
- **Training Notebook**: Complete Jupyter notebook for model training

## Performance

| Metric | Result |
|--------|--------|
| Model Accuracy | 93.8% overall |
| Qualified Image Accuracy | 100% |
| Terrain Classification | 99.97% confidence |
| Training Time | 1.26 hours (Tesla T4 GPU) |
| L1 Error | 0.0667 ± 0.0128 |

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/pix2pix-rekognition.git
cd pix2pix-rekognition

# Install dependencies
pip install -r requirements.txt

# Setup environment variables
cp .env.example .env
# Edit .env with your AWS credentials

# Run server
python app.py
```

### Training Your Own Model

1. Open `Pix2Pix_Satellite_to_Map.ipynb` in Google Colab
2. Runtime → Change runtime type → GPU (T4)
3. Runtime → Run all
4. Download trained model and place as `final_model.pth`

## API Usage

### Health Check
```bash
curl http://localhost:8000/health
```

### Generate Map (Original)
```bash
curl -X POST http://localhost:8000/generate \
  -F "image=@satellite_image.jpg"
```

### Generate Map (With Rekognition)
```bash
curl -X POST http://localhost:8000/generate-enhanced \
  -F "image=@satellite_image.jpg"
```

## Configuration

Edit `.env` file:

```env
AWS_REGION=us-east-1
AWS_ACCESS_KEY_ID=your_key_here
AWS_SECRET_ACCESS_KEY=your_secret_here
MODEL_PATH=final_model.pth
PORT=8000
```

## Architecture

- **Generator**: U-Net with skip connections (54.4M parameters)
- **Discriminator**: PatchGAN (2.8M parameters)
- **Dataset**: Berkeley Maps (2,194 paired images)
- **Training**: 90 epochs, ~1.26 hours on Tesla T4

## Project Structure

```
pix2pix-rekognition/
├── app.py                              # Flask API server
├── requirements.txt                    # Python dependencies
├── gunicorn_config.py                 # Production server config
├── final_model.pth                    # Trained model (not in git)
├── Pix2Pix_Satellite_to_Map.ipynb    # Training notebook
├── test_generate.py                   # API test script
└── .env                               # Environment variables (not in git)
```

## Testing

```bash
# Run test script
python test_generate.py

# Or test manually
curl http://localhost:8000/health
```

## Deployment

### Docker
```bash
docker build -t pix2pix-rekognition .
docker run -p 8000:8000 -e AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID -e AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY pix2pix-rekognition
```

### Production
```bash
gunicorn -c gunicorn_config.py app:app
```

## Cost Analysis

- **Pix2Pix Only**: $0.52 per 1,000 images
- **With Rekognition**: $1.95 per 1,000 images
- Quality filtering justifies additional cost

## Author

**Nguyen Thi Hong Linh**  

## References

- [Pix2Pix Paper](https://arxiv.org/abs/1611.07004) - Isola et al., 2017
- [AWS Rekognition Documentation](https://docs.aws.amazon.com/rekognition/)
- [Berkeley Maps Dataset](http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/)

