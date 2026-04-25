import json
import base64
from PIL import Image
import io

print("Decoding generated map...")

try:
    with open('result.json', 'r') as f:
        result = json.load(f)
    
    if result.get('status') == 'success':
        print(f"✓ Success!")
        print(f"  Terrain: {result.get('terrain_type')}")
        print(f"  Confidence: {result.get('confidence'):.1f}%")
        print(f"  Processing time: {result.get('total_processing_time'):.2f}s")
        
        # Decode base64 image
        img_data = base64.b64decode(result['generated_image'])
        img = Image.open(io.BytesIO(img_data))
        
        # Save as PNG
        img.save('generated_map.png')
        print("✓ Generated map saved as: generated_map.png")
        
        # Show the image
        img.show()
    
    elif result.get('status') == 'filtered':
        print(f"⚠ Image was filtered: {result.get('reason')}")
        print(f"  Terrain: {result.get('terrain_type')}")
        print(f"  Confidence: {result.get('confidence'):.1f}%")
    
    else:
        print(f"Error: {result}")

except Exception as e:
    print(f"✗ Error: {e}")