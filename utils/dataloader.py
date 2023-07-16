import numpy as np
from PIL import Image
from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing.image import ImageDataGenerator


class DataGenerator(Sequence):
    def __init__(self, image_filenames, mask_filenames, batch_size, target_size, augmentations=True):
        self.image_filenames = image_filenames
        self.mask_filenames = mask_filenames
        self.batch_size = batch_size
        self.target_size = target_size
        self.augmentations = augmentations

        self.image_data_generator = ImageDataGenerator(
            rotation_range=10,
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True,
            vertical_flip=True
        )

    def __len__(self):
        return int(np.ceil(len(self.image_filenames) / self.batch_size))

    def __getitem__(self, idx):
        batch_image_filenames = self.image_filenames[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_mask_filenames = self.mask_filenames[idx * self.batch_size:(idx + 1) * self.batch_size]

        batch_images = []
        batch_masks = []

        for image_filename, mask_filename in zip(batch_image_filenames, batch_mask_filenames):
            image = Image.open(image_filename)
            mask = Image.open(mask_filename).convert('L')  # Convert to grayscale

            image = image.resize(self.target_size)
            mask = mask.resize(self.target_size)

            image = np.array(image) / 255.0
            mask = np.array(mask) / 255.0

            if self.augmentations:
                seed = np.random.randint(1e6)
                image = self.apply_augmentations(image, seed)
                mask = self.apply_augmentations(mask, seed)
            else:
                mask = self.add_dim(mask)
            batch_images.append(image)
            batch_masks.append(mask)

        batch_images = np.array(batch_images)
        batch_masks = np.array(batch_masks)

        return batch_images, batch_masks

    def add_dim(self, item):
        if item.ndim == 2:  # Single-channel image
            item = np.expand_dims(item, axis=-1)
        return item

    def apply_augmentations(self, item, seed):
        if item.ndim == 2:  # Single-channel image
            item = np.expand_dims(item, axis=-1)
        augmented_item = self.image_data_generator.random_transform(item, seed=seed)
        if item.ndim == 3 and augmented_item.ndim == 4:
            augmented_item = np.squeeze(augmented_item, axis=0)
        return augmented_item


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    train_image_filenames = [r'../data/test/images/0a1ea1af4.jpg']
    train_mask_filenames = [r'../data/test/masks/0a1ea1af4.jpeg']

    batch_size = 2
    image_size = (512, 512)

    train_data_loader = DataGenerator(train_image_filenames, train_mask_filenames, batch_size, target_size=image_size,
                                      augmentations=True)

    print("Data sets loaded.")

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    batch_images, batch_masks = train_data_loader[0]
    print(batch_masks[0].max())
    print(batch_masks[0].min())

    axs[0].imshow(batch_images[0])
    axs[0].set_title('Image')

    axs[1].imshow(batch_masks[0], cmap='gray')
    axs[1].set_title('Mask')

    plt.tight_layout()
    plt.show()
