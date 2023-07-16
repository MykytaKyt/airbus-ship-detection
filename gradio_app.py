import gradio as gr
import numpy as np
from PIL import Image
import click
from tensorflow.keras.models import load_model


@click.command()
@click.option("--model_path", default="exp/model_final.h5", help="Path to the model file")
@click.option("--image_size", default=256, help="Size of the input image (both width and height)")
@click.option('--thres', type=float, default=0.5, help='Threshold of mask')
def main(model_path, image_size, thres):
    model = load_model(model_path, compile=False)

    def segment_image(input_image):
        # Resize and preprocess the input image
        image = Image.fromarray(input_image)
        image = image.convert('RGB')
        image = image.resize((image_size, image_size))
        input_data = np.array(image) / 255.0
        input_data = np.expand_dims(input_data, axis=0)

        output_masks = model.predict(input_data)
        output_mask = np.where(output_masks > thres, 1, 0)
        output_mask_2d = (output_mask * 255).astype(np.uint8)[0, :, :, 0]
        segmented_image = Image.fromarray(output_mask_2d, mode='L')

        return segmented_image

    def image_segmentation(input_image):
        segmented_image = segment_image(input_image)
        return input_image, segmented_image

    iface = gr.Interface(
        fn=image_segmentation,
        inputs=gr.inputs.Image(),
        outputs=["image", "image"]
    )

    iface.launch()


if __name__ == "__main__":
    main()
