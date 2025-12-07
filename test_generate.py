import requests
import base64
from PIL import Image
import io

print("Testing Pix2Pix Service...")

# Create a simple test image (or use an existing one)
print("1. Creating test image...")
test_img = Image.new('RGB', (256, 256))
# Create a gradient pattern
pixels = test_img.load()
for i in range(256):
    for j in range(256):
        pixels[j, i] = (i, j, (i+j)//2)

# Save as JPEG
buffered = io.BytesIO()
test_img.save(buffered, format="JPEG")
buffered.seek(0)

# Send to service
print("2. Sending to service...")
files = {'image': ('test.jpg', buffered.getvalue(), 'image/jpeg')}
response = requests.post('http://localhost:5000/generate', files=files, timeout=30)

print(f"3. Status Code: {response.status_code}")

if response.status_code == 200:
    result = response.json()
    print(f"4. Response Status: {result['status']}")
    
    if result['status'] == 'success':
        print(f"   Input Size: {result['input_size']}")
        print(f"   Output Size: {result['output_size']}")
        
        # Decode and save generated image
        img_data = base64.b64decode(result['generated_image'])
        output = Image.open(io.BytesIO(img_data))
        output.save('generated_map.png')
        
        print("✓ SUCCESS! Generated map saved as 'generated_map.png'")
        print("\nYour service is working perfectly!")
    else:
        print(f"✗ Error: {result['message']}")
else:
    print(f"✗ Request failed: {response.text}")