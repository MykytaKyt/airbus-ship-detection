filter:
	python3 filter.py --input-file '../../data/preprocessed/data.parquet' --output-file '../../data/preprocessed/filtered.parquet'
clean:
	python3 deduplication.py --input-file '../../data/preprocessed/filtered.parquet' --image-folder '../../data/train_v2/' --output-file '../../data/preprocessed/cleaned.parquet'
split:
	python3 train_test_split.py --input-file '../../data/preprocessed/cleaned.parquet' --image-folder '../../data/train_v2/' --test-size 0.2 --output-file '../../data/preprocessed/spitted.parquet'
create:
	python3 dataset.py --dataset-file '../../data/preprocessed/spitted.parquet'  --image-folder '../../data/train_v2/' --output-folder '../../data/'