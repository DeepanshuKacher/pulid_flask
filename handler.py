import numpy as np
import torch
from PIL import Image
import base64
from io import BytesIO
from pulid import attention_processor as attention
from pulid.pipeline_v1_1 import PuLIDPipeline
from pulid.utils import resize_numpy_image_long
from flask import Flask, request, jsonify

# import runpod
app = Flask(__name__)
app.config["DEBUG"] = True  # Enables debug mode globally

torch.set_grad_enabled(False)

# Initialize pipeline
pipeline = PuLIDPipeline(sdxl_repo='RunDiffusion/Juggernaut-XL-v9', sampler='dpmpp_2m')

DEFAULT_NEGATIVE_PROMPT = (
    'flaws in the eyes, flaws in the face, flaws, lowres, non-HDRi, low quality, worst quality,'
    'artifacts noise, text, watermark, glitch, deformed, mutated, ugly, disfigured, hands, '
    'low resolution, partially rendered objects,  deformed or partially rendered eyes, '
    'deformed, deformed eyeballs, cross-eyed,blurry'
)

def base64_to_numpy(base64_string):
    """
    Convert a base64 image string to a NumPy array.
    
    Parameters:
        base64_string (str): The base64-encoded image string.
    
    Returns:
        numpy.ndarray: The image as a NumPy array, or None if input is not provided.
    """
    if not base64_string:  # Check if input is present
        return None

    try:
        # Decode the base64 string
        image_data = base64.b64decode(base64_string)
        # Read the image data into a PIL Image
        image = Image.open(BytesIO(image_data))
        # Convert the PIL Image to a NumPy array
        numpy_array = np.array(image)
        return numpy_array
    except Exception as e:
        print(f"Error converting base64 string to NumPy array: {e}")
        return None

def numpy_to_base64(image_array: np.ndarray, image_format: str = "PNG") -> str:
    """
    Converts a NumPy array representing an image to a Base64-encoded string.

    Parameters:
        image_array (np.ndarray): The NumPy array representing the image.
        image_format (str): The format of the image (e.g., "PNG", "JPEG").

    Returns:
        str: Base64-encoded string of the image.
    """
    # Convert the NumPy array to a PIL Image
    image = Image.fromarray(image_array)

    # Save the image to a BytesIO buffer in the specified format
    buffer = BytesIO()
    image.save(buffer, format=image_format)
    buffer.seek(0)

    # Encode the buffer content to Base64
    base64_encoded = base64.b64encode(buffer.read())

    # Convert Base64 bytes to a string and return
    return base64_encoded.decode('utf-8')

@torch.inference_mode()
def run(*args):
    id_image = args[0]

    supp_images = args[1:4]
    prompt, neg_prompt, scale, seed, steps, H, W, id_scale, num_zero, ortho = args[4:]
    seed = int(seed)
    if seed == -1:
        seed = torch.Generator(device="cpu").seed()


    pipeline.debug_img_list = []

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


    if id_image is not None:
        id_image = resize_numpy_image_long(id_image, 1024)
        supp_id_image_list = [
            resize_numpy_image_long(supp_id_image, 1024) for supp_id_image in supp_images if supp_id_image is not None
        ]
        id_image_list = [id_image] + supp_id_image_list
        uncond_id_embedding, id_embedding = pipeline.get_id_embedding(id_image_list)
    else:
        uncond_id_embedding = None
        id_embedding = None
    

    img = pipeline.inference(
        prompt, (1, H, W), neg_prompt, id_embedding, uncond_id_embedding, id_scale, scale, steps, seed
    )[0]

    np_image  = np.array(img)

    base64_image = numpy_to_base64(np_image)

    return base64_image, str(seed)

def handler(event):
    data = event['input']
  
    image1 = base64_to_numpy(data.get('image1'))
    image2 = base64_to_numpy(data.get('image2'))
    image3 = base64_to_numpy(data.get('image3'))
    image4 = base64_to_numpy(data.get('image4'))
    prompt = data.get('prompt')

    if not prompt:
        raise Exception('Please provide a prompt')
    
    if not image1:
        raise Exception('Please provide at least one image')

 
    neg_prompt = data.get('neg_prompt', DEFAULT_NEGATIVE_PROMPT)
    scale = float(data.get('scale', 7.0))
    seed = int(data.get('seed', -1))
    steps = int(data.get('steps', 25))
    H = int(data.get('H', 1152))
    W = int(data.get('W', 896))
    id_scale = float(data.get('id_scale', 0.8))
    num_zero = int(data.get('num_zero', 20))
    ortho = data.get('ortho', 'v2')

    inps = [
        image1,
        image2,
        image3,
        image4,
        prompt,
        neg_prompt,
        scale,
        seed,
        steps,
        H,
        W,
        id_scale,
        num_zero,
        ortho,
    ]

    output, seed_output  = run(*inps)
    
    return {
        'output': output,
       'seed_output': seed_output,
    }


@app.route('/generate', methods=['POST'])
def generate():
    try:
        # Get input JSON data
        data = request.json
            
        image1 = base64_to_numpy(data.get('image1'))
        image2 = base64_to_numpy(data.get('image2'))
        image3 = base64_to_numpy(data.get('image3'))
        image4 = base64_to_numpy(data.get('image4'))
        prompt = data.get('prompt')

        if not prompt:
            raise Exception('Please provide a prompt')
        
        if not image1:
            raise Exception('Please provide at least one image')

    
        neg_prompt = data.get('neg_prompt', DEFAULT_NEGATIVE_PROMPT)
        scale = float(data.get('scale', 7.0))
        seed = int(data.get('seed', -1))
        steps = int(data.get('steps', 25))
        H = int(data.get('H', 1152))
        W = int(data.get('W', 896))
        id_scale = float(data.get('id_scale', 0.8))
        num_zero = int(data.get('num_zero', 20))
        ortho = data.get('ortho', 'v2')

        inps = [
            image1,
            image2,
            image3,
            image4,
            prompt,
            neg_prompt,
            scale,
            seed,
            steps,
            H,
            W,
            id_scale,
            num_zero,
            ortho,
        ]

        output, seed_output  = run(*inps)

        return jsonify({
            "output": output,  # Base64-encoded image
            "seed_output": seed_output,
            # "intermediate_output": intermediate_output  # Debug images if needed
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 400



# if __name__ == '__main__':
#     runpod.serverless.start({'handler': handler})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7860)
