import os
import click
import pandas as pd
import fiftyone as fo
import fiftyone.brain as fob


@click.command()
@click.option('--input-file', '-i', type=click.Path(exists=True), help='Input .parquet file path')
@click.option('--image-folder', '-f', type=click.Path(exists=True), help='Image folder path')
@click.option('--test-size', '-t', type=float, default=0.15, show_default=True, help='Test size (0.0 to 1.0)')
@click.option('--output-file', '-o', type=str, help='Path to the output splitted.parquet file')
def split_dataset(input_file, image_folder, test_size, output_file):
    data = pd.read_parquet(input_file)

    images = [image_folder + '/' + filename for filename in data['ImageId']]

    dataset = fo.Dataset.from_images(images)
    click.secho("Beginning compute uniqueness", fg='green')
    try:
        fob.compute_uniqueness(dataset)
        click.secho("Calculated successfully!", fg='green')
        uniqueness = []
        for sample in dataset:
            sample_info = {
                'ImageId': sample.filepath,
                'uniqueness': sample.uniqueness,
            }
            uniqueness.append(sample_info)
    except Exception as e:
        click.secho(str(e), fg='red')
        raise

    df_uniqueness = pd.DataFrame(uniqueness)
    df_uniqueness['ImageId'] = df_uniqueness['ImageId'].apply(os.path.basename)

    df_merged = pd.merge(data, df_uniqueness, on='ImageId', how='left')

    df_sorted = df_merged.sort_values('uniqueness', ascending=False).reset_index(drop=True)

    test_indices = df_sorted.head(int(test_size * len(df_sorted))).index
    train_indices = df_sorted.index.difference(test_indices)

    df_sorted.loc[test_indices, 'set'] = 'test'
    df_sorted.loc[train_indices, 'set'] = 'train'

    df_sorted.to_parquet(output_file, index=False)
    dataset.delete()
    click.secho("Splitting completed successfully!", fg='green')
    num_train_images = len(train_indices)
    num_test_images = len(test_indices)
    click.secho(f"Number of images in the train set: {num_train_images}", fg='green')
    click.secho(f"Number of images in the test set: {num_test_images}", fg='green')


if __name__ == "__main__":
    split_dataset()
