"""
Pix2Pix Satellite-to-Map Translation with AWS Rekognition Integration
=====================================================================

A Flask-based REST API service that translates satellite imagery to map representations
using a trained Pix2Pix GAN model, enhanced with AWS Rekognition for intelligent 
preprocessing and terrain classification.

Key Features:
    - Satellite-to-map image translation using U-Net Generator
    - Automated terrain classification (urban, rural, water, mixed)
    - Image quality filtering via AWS Rekognition
    - RESTful API with multiple endpoints
    - Environment-based configuration

Author: Linh Thi Hong Nguyen
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
import io
import base64
import logging
import os
import time
import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Flask application setup
app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Device configuration (CPU for EC2 compatibility)
device = torch.device('cpu')


# ==================== Model Architecture ====================

class UNetDown(nn.Module):
    """
    Downsampling block for U-Net encoder.
    
    Applies: Conv2d → (BatchNorm) → LeakyReLU → (Dropout)
    Reduces spatial dimensions by 2x while increasing feature channels.
    
    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        normalize (bool): Whether to apply batch normalization
        dropout (float): Dropout probability (0.0 = no dropout)
    """
    def __init__(self, in_channels, out_channels, normalize=True, dropout=0.0):
        super(UNetDown, self).__init__()
        layers = [nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False)]
        if normalize:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)


class UNetUp(nn.Module):
    """
    Upsampling block for U-Net decoder.
    
    Applies: ConvTranspose2d → BatchNorm → ReLU → (Dropout)
    Increases spatial dimensions by 2x and concatenates skip connections.
    
    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        dropout (float): Dropout probability (0.0 = no dropout)
    """
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super(UNetUp, self).__init__()
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)
    
    def forward(self, x, skip_input):
        """Forward pass with skip connection concatenation."""
        x = self.model(x)
        return torch.cat((x, skip_input), dim=1)


class GeneratorUNet(nn.Module):
    """
    U-Net Generator for Pix2Pix image-to-image translation.
    
    Architecture: 8-layer encoder → bottleneck → 7-layer decoder
    Total parameters: 54,413,955
    
    The U-Net architecture preserves spatial information through skip connections,
    allowing the model to generate high-quality maps from satellite imagery.
    
    Args:
        in_channels (int): Input image channels (default: 3 for RGB)
        out_channels (int): Output image channels (default: 3 for RGB)
        ngf (int): Number of generator filters in first layer (default: 64)
    """
    def __init__(self, in_channels=3, out_channels=3, ngf=64):
        super(GeneratorUNet, self).__init__()
        
        # Encoder: progressively downsample
        self.down1 = UNetDown(in_channels, ngf, normalize=False)
        self.down2 = UNetDown(ngf, ngf * 2)
        self.down3 = UNetDown(ngf * 2, ngf * 4)
        self.down4 = UNetDown(ngf * 4, ngf * 8)
        self.down5 = UNetDown(ngf * 8, ngf * 8)
        self.down6 = UNetDown(ngf * 8, ngf * 8)
        self.down7 = UNetDown(ngf * 8, ngf * 8)
        self.down8 = UNetDown(ngf * 8, ngf * 8, normalize=False)
        
        # Decoder: progressively upsample with skip connections
        self.up1 = UNetUp(ngf * 8, ngf * 8, dropout=0.5)
        self.up2 = UNetUp(ngf * 8 * 2, ngf * 8, dropout=0.5)
        self.up3 = UNetUp(ngf * 8 * 2, ngf * 8, dropout=0.5)
        self.up4 = UNetUp(ngf * 8 * 2, ngf * 8)
        self.up5 = UNetUp(ngf * 8 * 2, ngf * 4)
        self.up6 = UNetUp(ngf * 4 * 2, ngf * 2)
        self.up7 = UNetUp(ngf * 2 * 2, ngf)
        
        # Final layer: output RGB image
        self.final = nn.Sequential(
            nn.ConvTranspose2d(ngf * 2, out_channels, 4, 2, 1),
            nn.Tanh()
        )
    
    def forward(self, x):
        """
        Forward pass through U-Net.
        
        Args:
            x: Input tensor [batch, 3, 256, 256]
            
        Returns:
            Generated image tensor [batch, 3, 256, 256]
        """
        # Encoder path
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)
        
        # Decoder path with skip connections
        u1 = self.up1(d8, d7)
        u2 = self.up2(u1, d6)
        u3 = self.up3(u2, d5)
        u4 = self.up4(u3, d4)
        u5 = self.up5(u4, d3)
        u6 = self.up6(u5, d2)
        u7 = self.up7(u6, d1)
        
        return self.final(u7)


# ==================== AWS Rekognition Integration ====================

class RekognitionAnalyzer:
    """
    AWS Rekognition integration for intelligent image preprocessing.
    
    Provides automated terrain classification, quality filtering, and
    feature detection for satellite imagery before Pix2Pix processing.
    
    Features:
        - Label detection for terrain classification
        - Text detection for annotation identification
        - Confidence-based quality filtering
        - Terrain type classification (urban, rural, water, mixed)
    """
    
    def __init__(self):
        """Initialize AWS Rekognition client with IAM role authentication."""
        try:
            region = os.getenv('AWS_REGION', 'us-east-1')
            self.client = boto3.client('rekognition', region_name=region)
            logger.info("✓ AWS Rekognition client initialized successfully")
        except Exception as e:
            logger.error(f"✗ Failed to initialize Rekognition: {e}")
            self.client = None
    
    def analyze_image(self, image_bytes):
        """
        Analyze image using AWS Rekognition.
        
        Args:
            image_bytes: Raw image bytes
            
        Returns:
            dict: Analysis results containing labels and detected text
        """
        if not self.client:
            return {'error': 'Rekognition client not initialized'}
        
        try:
            # Detect labels (objects, scenes, concepts)
            labels_response = self.client.detect_labels(
                Image={'Bytes': image_bytes},
                MaxLabels=10,
                MinConfidence=50
            )
            
            results = {
                'labels': [
                    {
                        'name': l['Name'],
                        'confidence': l['Confidence'],
                        'categories': [c['Name'] for c in l.get('Categories', [])]
                    }
                    for l in labels_response['Labels']
                ]
            }
            
            # Detect text (optional, for annotation identification)
            try:
                text_response = self.client.detect_text(
                    Image={'Bytes': image_bytes}
                )
                results['text'] = [
                    {
                        'text': t['DetectedText'],
                        'confidence': t['Confidence'],
                        'type': t['Type']
                    }
                    for t in text_response['TextDetections']
                    if t['Type'] == 'LINE'
                ]
            except:
                results['text'] = []
            
            return results
            
        except ClientError as e:
            logger.error(f"Rekognition API error: {e}")
            return {'error': str(e)}
    
    def classify_terrain_type(self, analysis_results):
        """
        Classify terrain type based on detected labels.
        
        Uses keyword matching to categorize imagery into:
        - urban: Cities, buildings, roads
        - rural: Farms, fields, vegetation
        - water: Oceans, lakes, rivers
        - mixed: Multiple terrain types
        - unknown: Insufficient information
        
        Args:
            analysis_results (dict): Results from analyze_image()
            
        Returns:
            str: Terrain type classification
        """
        if 'error' in analysis_results:
            return 'unknown'
        
        labels = [l['name'].lower() for l in analysis_results.get('labels', [])]
        
        # Define terrain keywords
        urban_keywords = [
            'city', 'urban', 'building', 'architecture', 'downtown',
            'road', 'street', 'intersection', 'highway'
        ]
        rural_keywords = [
            'rural', 'farm', 'field', 'agriculture', 'countryside',
            'vegetation', 'forest', 'tree'
        ]
        water_keywords = [
            'water', 'ocean', 'sea', 'lake', 'river', 'bay', 'coast'
        ]
        
        # Calculate terrain scores
        urban_score = sum(1 for kw in urban_keywords if any(kw in label for label in labels))
        rural_score = sum(1 for kw in rural_keywords if any(kw in label for label in labels))
        water_score = sum(1 for kw in water_keywords if any(kw in label for label in labels))
        
        # Determine dominant terrain type
        scores = [(water_score, 'water'), (urban_score, 'urban'), (rural_score, 'rural')]
        scores.sort(reverse=True)
        
        if scores[0][0] >= 2:
            return scores[0][1]
        elif scores[0][0] >= 1 and scores[1][0] >= 1:
            return 'mixed'
        
        return 'unknown'
    
    def should_process_image(self, analysis_results):
        """
        Determine if image is suitable for Pix2Pix processing.
        
        Filters out non-aerial imagery (indoor scenes, portraits, etc.)
        and low-quality images based on confidence scores.
        
        Args:
            analysis_results (dict): Results from analyze_image()
            
        Returns:
            tuple: (should_process, reason, confidence)
        """
        if 'error' in analysis_results:
            return False, "Rekognition analysis failed", 0.0
        
        labels = analysis_results.get('labels', [])
        if not labels:
            return False, "No labels detected", 0.0
        
        # Check for aerial/outdoor imagery
        aerial_keywords = [
            'aerial', 'landscape', 'nature', 'outdoors', 'scenery',
            'land', 'terrain', 'topography'
        ]
        aerial_labels = [
            l for l in labels
            if any(kw in l['name'].lower() for kw in aerial_keywords)
        ]
        
        if not aerial_labels:
            return False, "Not an aerial/outdoor image", 0.0
        
        # Check confidence threshold
        max_confidence = max([l['confidence'] for l in aerial_labels])
        if max_confidence < 60:
            return False, f"Low confidence aerial image ({max_confidence:.1f}%)", max_confidence
        
        return True, "Suitable for processing", max_confidence


# ==================== Service Setup ====================

# Image preprocessing transform
transform = transforms.Compose([
    transforms.Resize((256, 256), transforms.InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load model from environment variable or default path
MODEL_PATH = os.getenv('MODEL_PATH', 'final_model.pth')

logger.info(f"Loading Pix2Pix model from: {MODEL_PATH}")
MODEL_LOADED = False
generator = None
rekognition_analyzer = None

try:
    generator = GeneratorUNet(in_channels=3, out_channels=3, ngf=64).to(device)
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    generator.load_state_dict(checkpoint['generator'])
    generator.eval()
    MODEL_LOADED = True
    logger.info("✓ Pix2Pix model loaded successfully")
    
    rekognition_analyzer = RekognitionAnalyzer()
    
except Exception as e:
    logger.error(f"✗ Initialization error: {e}")


# ==================== API Endpoints ====================

@app.route('/')
def home():
    """
    Service information endpoint.
    
    Returns:
        JSON with service status and available endpoints
    """
    return jsonify({
        "service": "Pix2Pix Satellite-to-Map Translation",
        "version": "2.0.0",
        "status": "online",
        "model_loaded": MODEL_LOADED,
        "rekognition_enabled": rekognition_analyzer is not None and rekognition_analyzer.client is not None,
        "author": "Nguyen Thi Hong Linh",
        "description": "Satellite-to-map translation with AWS Rekognition preprocessing",
        "endpoints": {
            "GET /": "Service information",
            "GET /health": "Health check",
            "GET /info": "Model details",
            "POST /generate": "Generate map (standard endpoint)",
            "POST /generate-enhanced": "Generate map with Rekognition analysis"
        }
    })


@app.route('/health')
def health():
    """
    Health check endpoint for monitoring and load balancing.
    
    Returns:
        JSON with service health status
    """
    return jsonify({
        "status": "healthy" if MODEL_LOADED else "unhealthy",
        "model_loaded": MODEL_LOADED,
        "rekognition_enabled": rekognition_analyzer is not None and rekognition_analyzer.client is not None,
        "device": str(device),
        "version": "2.0.0"
    })


@app.route('/info')
def info():
    """
    Model information endpoint.
    
    Returns:
        JSON with model architecture details and capabilities
    """
    if not MODEL_LOADED:
        return jsonify({"error": "Model not loaded"}), 503
    
    return jsonify({
        "model_name": "Pix2Pix GAN with AWS Rekognition",
        "architecture": "U-Net Generator + PatchGAN Discriminator",
        "parameters": "54,413,955",
        "input_size": "256×256 RGB",
        "output_size": "256×256 RGB",
        "aws_service": "Amazon Rekognition",
        "features": [
            "Terrain classification (urban/rural/water)",
            "Automated quality filtering",
            "Feature detection and labeling"
        ]
    })


@app.route('/generate', methods=['POST'])
def generate():
    """
    Standard image generation endpoint (Assignment 5 compatibility).
    
    Accepts satellite imagery and generates corresponding map representation
    without Rekognition preprocessing.
    
    Request:
        - Form data with 'image' file, or
        - JSON with base64-encoded 'image' field
        
    Returns:
        JSON with:
        - status: success/error
        - generated_image: Base64-encoded PNG
        - input_size: Original image dimensions
        - output_size: Generated image dimensions [256, 256]
        - message: Status message
    """
    if not MODEL_LOADED:
        return jsonify({
            "status": "error",
            "message": "Model not loaded"
        }), 503
    
    try:
        # Parse input image
        image_bytes = None
        if 'image' in request.files:
            image_bytes = request.files['image'].read()
        elif request.is_json and 'image' in request.json:
            image_bytes = base64.b64decode(request.json['image'])
        else:
            return jsonify({
                "status": "error",
                "message": "No image provided. Send as form-data or JSON."
            }), 400
        
        # Load and preprocess image
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        original_size = image.size
        input_tensor = transform(image).unsqueeze(0).to(device)
        
        # Generate map
        with torch.no_grad():
            output_tensor = generator(input_tensor)
        
        # Post-process output
        output_tensor = output_tensor.squeeze(0).cpu() * 0.5 + 0.5
        output_tensor = torch.clamp(output_tensor, 0, 1)
        output_np = output_tensor.permute(1, 2, 0).numpy()
        output_image = Image.fromarray((output_np * 255).astype('uint8'))
        
        # Encode as base64
        buffered = io.BytesIO()
        output_image.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        return jsonify({
            "status": "success",
            "generated_image": img_base64,
            "input_size": list(original_size),
            "output_size": [256, 256],
            "message": "Map generated successfully"
        })
        
    except Exception as e:
        logger.error(f"Generation error: {e}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


@app.route('/generate-enhanced', methods=['POST'])
def generate_enhanced():
    """
    Enhanced generation endpoint with AWS Rekognition preprocessing.
    
    Performs intelligent preprocessing before map generation:
    1. Analyzes image with AWS Rekognition
    2. Classifies terrain type
    3. Filters unsuitable imagery
    4. Generates map with metadata annotations
    
    Request:
        - Form data with 'image' file, or
        - JSON with base64-encoded 'image' field
        
    Returns:
        JSON with:
        - status: success/filtered/error
        - generated_image: Base64-encoded PNG (if successful)
        - terrain_type: Classified terrain category
        - confidence: Rekognition confidence score
        - rekognition_analysis: Detected labels and processing time
        - pix2pix_processing_time: Generation time (seconds)
        - total_processing_time: Total pipeline time (seconds)
        
    Note:
        Images failing quality checks return status='filtered' without generation.
    """
    if not MODEL_LOADED or not rekognition_analyzer or not rekognition_analyzer.client:
        return jsonify({
            "status": "error",
            "message": "Service not fully initialized"
        }), 503
    
    try:
        # Parse input image
        image_bytes = None
        if 'image' in request.files:
            image_bytes = request.files['image'].read()
        elif request.is_json and 'image' in request.json:
            image_bytes = base64.b64decode(request.json['image'])
        else:
            return jsonify({
                "status": "error",
                "message": "No image provided. Send as form-data or JSON."
            }), 400
        
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        original_size = image.size
        
        # Step 1: AWS Rekognition analysis
        rekog_start = time.time()
        rekognition_results = rekognition_analyzer.analyze_image(image_bytes)
        rekog_time = time.time() - rekog_start
        
        terrain_type = rekognition_analyzer.classify_terrain_type(rekognition_results)
        should_process, reason, confidence = rekognition_analyzer.should_process_image(rekognition_results)
        
        # Step 2: Quality filtering
        if not should_process:
            return jsonify({
                "status": "filtered",
                "reason": reason,
                "confidence": confidence,
                "terrain_type": terrain_type,
                "rekognition_analysis": {
                    "labels": rekognition_results.get('labels', [])[:5],
                    "processing_time": rekog_time,
                    "text_detected": len(rekognition_results.get('text', []))
                }
            })
        
        # Step 3: Pix2Pix generation
        pix2pix_start = time.time()
        input_tensor = transform(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output_tensor = generator(input_tensor)
        
        output_tensor = output_tensor.squeeze(0).cpu() * 0.5 + 0.5
        output_tensor = torch.clamp(output_tensor, 0, 1)
        output_np = output_tensor.permute(1, 2, 0).numpy()
        output_image = Image.fromarray((output_np * 255).astype('uint8'))
        pix2pix_time = time.time() - pix2pix_start
        
        # Step 4: Add metadata annotations
        annotated = output_image.copy()
        draw = ImageDraw.Draw(annotated)
        try:
            font = ImageFont.load_default()
            draw.text((10, 10), f"Terrain: {terrain_type.upper()}", fill=(255, 0, 0), font=font)
            for i, label in enumerate(rekognition_results.get('labels', [])[:3]):
                draw.text((10, 30 + i*15),
                         f"{label['name']}: {label['confidence']:.0f}%",
                         fill=(0, 0, 255), font=font)
        except Exception as e:
            logger.warning(f"Annotation error: {e}")
        
        # Encode output
        buffered = io.BytesIO()
        annotated.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        return jsonify({
            "status": "success",
            "generated_image": img_base64,
            "input_size": list(original_size),
            "output_size": [256, 256],
            "terrain_type": terrain_type,
            "confidence": confidence,
            "rekognition_analysis": {
                "labels": rekognition_results.get('labels', [])[:5],
                "processing_time": rekog_time,
                "text_detected": len(rekognition_results.get('text', []))
            },
            "pix2pix_processing_time": pix2pix_time,
            "total_processing_time": rekog_time + pix2pix_time,
            "message": f"Map generated successfully. Terrain: {terrain_type}"
        })
        
    except Exception as e:
        logger.error(f"Enhanced generation error: {e}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


# ==================== Application Entry Point ====================

if __name__ == '__main__':
    port = int(os.getenv('PORT', 8000))
    debug = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
    
    logger.info(f"Starting Flask application on port {port}")
    app.run(host='0.0.0.0', port=port, debug=debug)