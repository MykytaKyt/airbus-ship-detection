train:
	python3 train.py \
        --save_dir='./exp/' \
        --save_name='model' \
        --logs=logs \
        --epoch=20 \
        --batch_size=16 \
        --lr=1e-4 \
        --patience=3 \
        --weight_decay=5e-5 \
        --image_size=256,256 \
        --train_dir='data/train' \
        --test_dir='data/test'
infer:
	python3 inference.py \
		--model_path 'weights/model_final.h5' \
		--image_path 'data/test_image/0a1a7f395.jpg' \
		--output_path 'test.jpg' \
		--image_size 256,256 \
		--thres 0.5
demo:
	python3 gradio_app.py \
		--model_path 'weights/model_final.h5' \
		--image_size 256 \
		--thres 0.5
api:
	python app.py \
		--model-path 'weights/model_final.h5' \
 		--image-size 256 \
 		--thres 0.5
unit_tests:
	python -m unittest discover tests
load_test:
	locust -f tests/locust_load.py --host=http://127.0.0.1:8000
