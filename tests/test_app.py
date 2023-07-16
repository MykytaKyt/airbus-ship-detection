import unittest
import requests

URL = 'http://127.0.0.1:8000/segment'
IMAGE_PATH = './data/test_image/'
IMAGE_NAME = '0bf631128.jpg'
IMAGE_FORMAT = 'image/jpeg'


class ImageSegmentationTest(unittest.TestCase):
    def test_image_segmentation(self):
        with open(IMAGE_PATH+IMAGE_NAME, 'rb') as image_file:
            image_data = image_file.read()

        headers = {'accept': 'application/json'}

        files = {
            'file': (IMAGE_NAME, image_data, IMAGE_FORMAT)
        }

        response = requests.post(URL, headers=headers, files=files)

        self.assertEqual(response.status_code, 200)

        response_json = response.json()

        self.assertIn("segmented_image", response_json)

    def test_missing_image_file(self):
        headers = {'accept': 'application/json'}

        files = {}

        response = requests.post(URL, headers=headers, files=files)

        self.assertEqual(response.status_code, 422)


if __name__ == '__main__':
    unittest.main()
