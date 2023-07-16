import click
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model


@click.command()
@click.option('--model_path', type=str, required=True, help='Path to the trained model')
@click.option('--image_path', type=str, required=True, help='Path to the input image')
@click.option('--output_path', type=str, required=True, help='Path to save the output segmentation mask')
@click.option('--image_size', type=str, default='256,256', help='Image size (width,height)')
@click.option('--thres', type=float, default=0.5, help='Threshold of mask')
def inference(model_path, image_path, output_path, image_size, thres):
    image_size = tuple(map(int, image_size.split(',')))
    model = load_model(model_path, compile=False)

    image = Image.open(image_path).convert('RGB')
    image = image.resize(image_size)
    input_data = np.array(image) / 255.0
    input_data = np.expand_dims(input_data, axis=0)

    output_masks = model.predict(input_data)

    output_mask = np.where(output_masks > thres, 1, 0)
    output_mask_2d = (output_mask * 255).astype(np.uint8)[0, :, :, 0]
    if output_path:
        output_image = Image.fromarray(output_mask_2d, mode='L')
        output_image.save(output_path)
        click.secho('Inference completed! Segmentation mask saved at {}'.format(output_path), fg='green')
    return output_mask


if __name__ == '__main__':
    inference()
