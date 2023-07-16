from locust import HttpUser, task, between

RENDER_ENDPOINT = "/segment"
IMAGE_PATH = "./data/test_image/0a1a7f395.jpg"


class FastAPIUser(HttpUser):
    wait_time = between(1, 5)

    @task
    def segment(self):
        with open(IMAGE_PATH, "rb") as image_file:
            files = {"file": image_file}
            response = self.client.post(RENDER_ENDPOINT, files=files)

        assert response.status_code == 200
