import uvicorn
import numpy as np
from PIL import Image
from io import BytesIO
import base64
from fastapi import FastAPI, File, UploadFile
from tensorflow.keras.models import load_model
import click

app = FastAPI()


@click.command()
@click.option("--model-path", default="exp/model_final.h5", help="Path to the trained model")
@click.option("--image-size", default=256, help="Size of the input image")
@click.option('--thres', type=float, default=0.5, help='Threshold of mask')
@click.option("--host", default="0.0.0.0", help="Host for api")
@click.option("--port", default=8000, help="Port for api")
def main(model_path, image_size, thres, host, port):
    model = load_model(model_path, compile=False)

    def preprocess_image(input_image):
        input = Image.open(BytesIO(input_image.read()))
        image = input.convert("RGB")
        image = image.resize((image_size, image_size))
        input_data = np.array(image) / 255.0
        input_data = np.expand_dims(input_data, axis=0)
        return input_data

    def segment_image(input_data):
        output_masks = model.predict(input_data)
        output_mask = np.where(output_masks > thres, 1, 0)
        output_mask_2d = (output_mask * 255).astype(np.uint8)[0, :, :, 0]
        segmented_image = Image.fromarray(output_mask_2d, mode='L')
        return segmented_image

    def encode_image(image):
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        encoded_image = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return encoded_image

    @app.post("/segment")
    async def segment(file: UploadFile = File(...)):
        input_data = preprocess_image(file.file)
        segmented_image = segment_image(input_data)
        encoded_segmented_image = encode_image(segmented_image)
        return {"segmented_image": encoded_segmented_image}

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
