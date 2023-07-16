import os
import click
import pandas as pd
from click import secho
from shutil import copyfile
from PIL import Image, ImageDraw
from concurrent.futures import ThreadPoolExecutor


def rle_to_pixels(rle_code):
    rle_code = [int(i) for i in rle_code.split()]
    pixels = [(pixel_position // 768, pixel_position % 768)
              for start, length in list(zip(rle_code[0:-1:2], rle_code[1::2]))
              for pixel_position in range(start, start + length)]
    return pixels


@click.command()
@click.option('--dataset-file', '-d', type=str, help='Path to the splitted.parquet file')
@click.option('--image-folder', '-i', type=str, help='Input image folder path')
@click.option('--output-folder', '-o', type=str, help='Output dataset folder path')
def create_dataset(dataset_file, image_folder, output_folder):
    df = pd.read_parquet(dataset_file)
    train_images_folder = os.path.join(output_folder, 'train', 'images')
    train_masks_folder = os.path.join(output_folder, 'train', 'masks')
    test_images_folder = os.path.join(output_folder, 'test', 'images')
    test_masks_folder = os.path.join(output_folder, 'test', 'masks')

    # Create the necessary folders
    os.makedirs(train_images_folder, exist_ok=True)
    os.makedirs(train_masks_folder, exist_ok=True)
    os.makedirs(test_images_folder, exist_ok=True)
    os.makedirs(test_masks_folder, exist_ok=True)

    def process_row(row):
        image_id = row['ImageId']
        set_type = row['set']
        image_path = os.path.join(image_folder, image_id)

        image_filename = os.path.basename(image_path)
        destination_folder = train_images_folder if set_type == 'train' else test_images_folder
        destination_path = os.path.join(destination_folder, image_filename)
        copyfile(image_path, destination_path)

        masks = row['annotations']
        combined_mask = Image.new('1', (768, 768))
        for i, mask in enumerate(masks):
            mask_pixels = rle_to_pixels(mask)
            draw = ImageDraw.Draw(combined_mask)
            draw.point(mask_pixels, fill=1)

        mask_filename = f'{os.path.splitext(image_filename)[0]}.jpg'
        destination_folder = train_masks_folder if set_type == 'train' else test_masks_folder
        destination_path = os.path.join(destination_folder, mask_filename)
        combined_mask.save(destination_path, "JPG")

        secho(f'Processed {image_filename} and its masks.', fg='green')

    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = []
        for _, row in df.iterrows():
            future = executor.submit(process_row, row)
            futures.append(future)

        for future in futures:
            future.result()

    secho('Dataset creation completed.', fg='green')


if __name__ == "__main__":
    create_dataset()
