import os
import click
import shutil
import fastdup
import pandas as pd


def get_broken_images(fd):
    return fd.invalid_instances()['filename'].to_list()


def get_duplicates(fd):
    clusters_df = get_clusters(fd.connected_components()[0])
    list_of_duplicates = []

    for cluster_file_list in clusters_df.filename:
        list_of_duplicates.extend(cluster_file_list[1:])

    return list(set(list_of_duplicates))


def get_outliers(fd):
    return fd.outliers()[fd.outliers().distance < 0.68]['filename_outlier'].tolist()


def get_dark_images(stats_df, threshold=13):
    return stats_df[stats_df['mean'] < threshold]['filename'].to_list()


def get_bright_images(stats_df, threshold=220.5):
    return stats_df[stats_df['mean'] > threshold]['filename'].to_list()


def get_blurry_images(stats_df, threshold=50):
    return stats_df[stats_df['blur'] < threshold]['filename'].to_list()


def get_clusters(df, sort_by='count', min_count=2, ascending=False):
    agg_dict = {'filename': list, 'mean_distance': max, 'count': len}

    if 'label' in df.columns:
        agg_dict['label'] = list

    df = df[df['count'] >= min_count]
    grouped_df = df.groupby('component_id').agg(agg_dict)
    grouped_df = grouped_df.sort_values(by=[sort_by], ascending=ascending)

    return grouped_df


@click.command()
@click.option('--input-file', '-i', type=click.Path(exists=True), help='Input .parquet file path')
@click.option('--image-folder', '-f', type=click.Path(exists=True), help='Image folder path')
@click.option('--output-file', '-o', type=click.Path(), help='Output cleaned .parquet file path')
def main(input_file, image_folder, output_file):
    data = pd.read_parquet(input_file)
    images = [image_folder + '/' + filename for filename in data['ImageId']]
    work_dir = "fastdup_work_dir/"

    try:
        fd = fastdup.create(work_dir=work_dir, input_dir=images)
        fd.run(ccthreshold=0.99)

        broken_images = get_broken_images(fd)
        duplicates = get_duplicates(fd)
        outliers = get_outliers(fd)
        dark_images = get_dark_images(fd.img_stats())
        bright_images = get_bright_images(fd.img_stats())
        blurry_images = get_blurry_images(fd.img_stats())

        problem_images = broken_images + duplicates + outliers + dark_images + bright_images + blurry_images

        click.secho(f"Number of images before cleaning: {len(data)}", fg='yellow')
        click.secho(f"Broken: {len(broken_images)}", fg='yellow')
        click.secho(f"Duplicates: {len(duplicates)}", fg='yellow')
        click.secho(f"Outliers: {len(outliers)}", fg='yellow')
        click.secho(f"Dark: {len(dark_images)}", fg='yellow')
        click.secho(f"Bright: {len(bright_images)}", fg='yellow')
        click.secho(f"Blurry: {len(blurry_images)}", fg='yellow')

        problem_files = [os.path.basename(filepath) for filepath in problem_images]
        cleaned_data = data[~data['ImageId'].isin(problem_files)]
        cleaned_data.to_parquet(output_file, index=False)

        click.secho(f"Cleaned data saved to {output_file}", fg='green')
        click.secho(f"Number of images after cleaning: {len(cleaned_data)}", fg='green')


    finally:
        shutil.rmtree(work_dir)
        click.secho("Working directory deleted.", fg='green')


if __name__ == '__main__':
    main()
