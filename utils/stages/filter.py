import click
import pandas as pd


@click.command()
@click.option('--input-file', help='Path to the input Parquet file')
@click.option('--output-file', help='Path to save the filtered Parquet file')
def filter_and_save_parquet(input_file, output_file):
    df = pd.read_parquet(input_file)
    initial_row_count = len(df)
    filtered_df = df[df['polygon_exists'] == True]
    filtered_row_count = len(filtered_df)
    filtered_df.to_parquet(output_file, index=False)
    click.secho(f"Initial row count: {initial_row_count}", fg='green')
    click.secho(f"Filtered row count: {filtered_row_count}", fg='green')
    click.echo("Filtered data saved successfully!")


if __name__ == '__main__':
    filter_and_save_parquet()