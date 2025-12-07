"""
Pix2Pix Satellite-to-Map Generator - Flask API
Converted from Assignment 5 Jupyter Notebook
"""

# Step 1: Import necessary libraries
from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import io
import base64
import logging
import os

# Step 2: Setup Flask app
app = Flask(__name__)
CORS(app)  # Allow cross-origin requests

# Setup logging to see what's happening
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Device configuration (use CPU for deployment)
device = torch.device('cpu')

# ============================================================================
# Step 3: Copy Model Architecture from Your Notebook
# (Cells 16-18 from your notebook)
# ============================================================================

class UNetDown(nn.Module):
    """Downsampling block - copied from your notebook"""
    def __init__(self, in_channels, out_channels, normalize=True, dropout=0.0):
        super(UNetDown, self).__init__()
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False)]
        if normalize:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)


class UNetUp(nn.Module):
    """Upsampling block - copied from your notebook"""
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super(UNetUp, self).__init__()
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)
    
    def forward(self, x, skip_input):
        x = self.model(x)
        x = torch.cat((x, skip_input), dim=1)
        return x


class GeneratorUNet(nn.Module):
    """U-Net Generator - copied from your notebook (Cell 18)"""
    def __init__(self, in_channels=3, out_channels=3, ngf=64):
        super(GeneratorUNet, self).__init__()
        
        # Encoder (downsampling)
        self.down1 = UNetDown(in_channels, ngf, normalize=False)
        self.down2 = UNetDown(ngf, ngf * 2)
        self.down3 = UNetDown(ngf * 2, ngf * 4)
        self.down4 = UNetDown(ngf * 4, ngf * 8)
        self.down5 = UNetDown(ngf * 8, ngf * 8)
        self.down6 = UNetDown(ngf * 8, ngf * 8)
        self.down7 = UNetDown(ngf * 8, ngf * 8)
        self.down8 = UNetDown(ngf * 8, ngf * 8, normalize=False)
        
        # Decoder (upsampling)
        self.up1 = UNetUp(ngf * 8, ngf * 8, dropout=0.5)
        self.up2 = UNetUp(ngf * 8 * 2, ngf * 8, dropout=0.5)
        self.up3 = UNetUp(ngf * 8 * 2, ngf * 8, dropout=0.5)
        self.up4 = UNetUp(ngf * 8 * 2, ngf * 8)
        self.up5 = UNetUp(ngf * 8 * 2, ngf * 4)
        self.up6 = UNetUp(ngf * 4 * 2, ngf * 2)
        self.up7 = UNetUp(ngf * 2 * 2, ngf)
        
        # Final layer
        self.final = nn.Sequential(
            nn.ConvTranspose2d(ngf * 2, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )
    
    def forward(self, x):
        # Encoder
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)
        
        # Decoder with skip connections
        u1 = self.up1(d8, d7)
        u2 = self.up2(u1, d6)
        u3 = self.up3(u2, d5)
        u4 = self.up4(u3, d4)
        u5 = self.up5(u4, d3)
        u6 = self.up6(u5, d2)
        u7 = self.up7(u6, d1)
        
        return self.final(u7)


# ============================================================================
# Step 4: Setup Image Preprocessing (from Cell 11 in your notebook)
# ============================================================================

transform = transforms.Compose([
    transforms.Resize((256, 256), transforms.InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


# ============================================================================
# Step 5: Load Your Trained Model
# ============================================================================

logger.info("Loading trained model...")
MODEL_LOADED = False
generator = None

try:
    # Create generator instance
    generator = GeneratorUNet(in_channels=3, out_channels=3, ngf=64).to(device)
    
    # Load the trained weights
    checkpoint = torch.load('final_model.pth', map_location=device, weights_only=False)
    generator.load_state_dict(checkpoint['generator'])
    
    # Set to evaluation mode (important!)
    generator.eval()
    
    MODEL_LOADED = True
    logger.info("✓ Model loaded successfully!")
    
except FileNotFoundError:
    logger.error("✗ Error: final_model.pth not found! Make sure it's in the same folder as app.py")
except Exception as e:
    logger.error(f"✗ Error loading model: {e}")


# ============================================================================
# Step 6: Create Flask Endpoints
# ============================================================================

@app.route('/')
def home():
    """Home page - shows service info"""
    return jsonify({
        "service": "Pix2Pix Satellite-to-Map Generator",
        "status": "online",
        "model_loaded": MODEL_LOADED,
        "author": "Linh Nguyen",
        "description": "Converts satellite imagery to map-style visualizations using Pix2Pix GAN",
        "endpoints": {
            "GET /": "This page - service information",
            "GET /health": "Check if service is healthy",
            "GET /info": "Get model details",
            "POST /generate": "Generate map from satellite image"
        },
        "usage": "Send POST request with image to /generate endpoint"
    })


@app.route('/health')
def health():
    """Health check - for monitoring"""
    return jsonify({
        "status": "healthy" if MODEL_LOADED else "unhealthy",
        "model_loaded": MODEL_LOADED,
        "device": str(device),
        "version": "1.0.0"
    })


@app.route('/info')
def info():
    """Model information"""
    if not MODEL_LOADED:
        return jsonify({"error": "Model not loaded"}), 503
    
    return jsonify({
        "model_name": "Pix2Pix GAN",
        "architecture": "U-Net Generator with PatchGAN Discriminator",
        "parameters": "54,413,955",
        "input_size": "256x256 RGB",
        "output_size": "256x256 RGB",
        "training_epochs": 10,
        "dataset": "Maps dataset (satellite to map translation)",
        "framework": "PyTorch"
    })


@app.route('/generate', methods=['POST'])
def generate():
    """
    Main endpoint - generates map from satellite image
    
    Accepts:
    - multipart/form-data with 'image' file
    - JSON with base64 encoded 'image'
    
    Returns:
    - JSON with generated map as base64
    """
    
    # Check if model is loaded
    if not MODEL_LOADED:
        return jsonify({
            "status": "error",
            "message": "Model not loaded. Please check server logs."
        }), 503
    
    try:
        # Get image from request
        image_bytes = None
        
        if 'image' in request.files:
            # Method 1: File upload (multipart/form-data)
            file = request.files['image']
            image_bytes = file.read()
            logger.info(f"Received file: {file.filename}")
            
        elif request.is_json and 'image' in request.json:
            # Method 2: Base64 encoded in JSON
            image_b64 = request.json['image']
            image_bytes = base64.b64decode(image_b64)
            logger.info("Received base64 encoded image")
            
        else:
            return jsonify({
                "status": "error",
                "message": "No image provided. Send as 'image' in form-data or base64 in JSON body.",
                "examples": {
                    "curl": "curl -X POST [URL]/generate -F 'image=@photo.jpg'",
                    "json": '{"image": "<base64_string>"}'
                }
            }), 400
        
        # Load and validate image
        try:
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            original_size = image.size
            logger.info(f"Image loaded: {original_size}")
        except Exception as e:
            return jsonify({
                "status": "error",
                "message": f"Invalid image format. Please upload JPEG or PNG. Error: {str(e)}"
            }), 400
        
        # Preprocess image (same as training)
        input_tensor = transform(image).unsqueeze(0).to(device)
        logger.info(f"Image preprocessed: {input_tensor.shape}")
        
        # Generate map using your trained model
        with torch.no_grad():
            output_tensor = generator(input_tensor)
        
        # Post-process output
        output_tensor = output_tensor.squeeze(0).cpu()
        output_tensor = output_tensor * 0.5 + 0.5  # Denormalize from [-1,1] to [0,1]
        output_tensor = torch.clamp(output_tensor, 0, 1)
        
        # Convert to PIL Image
        output_np = output_tensor.permute(1, 2, 0).numpy()
        output_image = Image.fromarray((output_np * 255).astype('uint8'))
        
        # Encode to base64 for sending back
        buffered = io.BytesIO()
        output_image.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        logger.info("Generation complete!")
        
        # Return result
        return jsonify({
            "status": "success",
            "generated_image": img_base64,
            "input_size": list(original_size),
            "output_size": [256, 256],
            "message": "Map generated successfully from satellite image"
        })
    
    except Exception as e:
        logger.error(f"Error during generation: {e}")
        return jsonify({
            "status": "error",
            "message": f"Generation failed: {str(e)}"
        }), 500


# ============================================================================
# Step 7: Run the Flask App
# ============================================================================

if __name__ == '__main__':
    # Get port from environment variable (for deployment) or use 5000
    port = int(os.environ.get('PORT', 5000))
    
    # Run the app
    logger.info(f"Starting Flask app on port {port}...")
    app.run(host='0.0.0.0', port=port, debug=False)