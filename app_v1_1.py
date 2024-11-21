from flask import Flask, request, jsonify
import numpy as np
import torch
from PIL import Image
import base64
from io import BytesIO
from pulid import attention_processor as attention
from pulid.pipeline_v1_1 import PuLIDPipeline
from pulid.utils import resize_numpy_image_long

# Initialize Flask app
app = Flask(__name__)

# Torch settings
torch.set_grad_enabled(False)

# Initialize pipeline
pipeline = PuLIDPipeline(sdxl_repo='RunDiffusion/Juggernaut-XL-v9', sampler='dpmpp_2m')

# Constants
DEFAULT_NEGATIVE_PROMPT = (
    'flaws in the eyes, flaws in the face, flaws, lowres, non-HDRi, low quality, worst quality,'
    'artifacts noise, text, watermark, glitch, deformed, mutated, ugly, disfigured, hands, '
    'low resolution, partially rendered objects,  deformed or partially rendered eyes, '
    'deformed, deformed eyeballs, cross-eyed,blurry'
)

def parse_image(image_data):
    """Parse and decode base64 image data."""
    if not image_data:
        raise ValueError("Image data is required but not provided.")
    image_bytes = base64.b64decode(image_data)
    image = Image.open(BytesIO(image_bytes))
    return np.array(image)

def numpy_to_base64(image_array):
    """Converts a NumPy image array to a Base64-encoded string."""
    image = Image.fromarray(image_array)
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode('utf-8')

@app.route('/generate', methods=['POST'])
def generate():
    try:
        # Get input JSON data
        data = request.json
        
        # Parse images
        id_image = parse_image(data.get('id_image'))
        supp_images = [parse_image(data.get(f'supp_image{i}')) for i in range(1, 4)]
        supp_images = [img for img in supp_images if img is not None]

        # Extract other parameters
        prompt = data.get('prompt', 'portrait,color,cinematic,in garden,soft light,detailed face')
        neg_prompt = data.get('neg_prompt', DEFAULT_NEGATIVE_PROMPT)
        scale = float(data.get('scale', 7.0))
        seed = int(data.get('seed', -1))
        steps = int(data.get('steps', 25))
        H = int(data.get('H', 1152))
        W = int(data.get('W', 896))
        id_scale = float(data.get('id_scale', 0.8))
        num_zero = int(data.get('num_zero', 20))
        ortho = data.get('ortho', 'v2')

        # Set up attention settings
        attention.NUM_ZERO = num_zero
        if ortho == 'v2':
            attention.ORTHO = False
            attention.ORTHO_v2 = True
        elif ortho == 'v1':
            attention.ORTHO = True
            attention.ORTHO_v2 = False
        else:
            attention.ORTHO = False
            attention.ORTHO_v2 = False

        # Process ID image embeddings
        if id_image is not None:
            id_image = resize_numpy_image_long(id_image, 1024)
            supp_id_image_list = [
                resize_numpy_image_long(supp_id_image, 1024) for supp_id_image in supp_images
            ]
            id_image_list = [id_image] + supp_id_image_list
            uncond_id_embedding, id_embedding = pipeline.get_id_embedding(id_image_list)
        else:
            uncond_id_embedding = None
            id_embedding = None

        # Seed handling
        if seed == -1:
            seed = torch.Generator(device="cpu").seed()

        # Inference
        img = pipeline.inference(
            prompt, (1, H, W), neg_prompt, id_embedding, uncond_id_embedding, id_scale, scale, steps, seed
        )[0]

        # Convert image to Base64
        image_base64 = numpy_to_base64(img)

        return jsonify({
            "output": image_base64,  # Base64-encoded image
            "seed_output": str(seed),
            "intermediate_output": pipeline.debug_img_list  # Debug images if needed
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=7860)
