import click
import os
import tensorflow as tf
import glob
from utils.loss.weighted_bce_dice_loss import weighted_bce_dice_loss
from utils.model.unet import build_resnet50_unet
from utils.dataloader import DataGenerator
from utils.metrics import dice_coef
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, ReduceLROnPlateau
from tensorflow.keras.metrics import MeanIoU, Accuracy
import tensorflow_addons as tfa

@click.command()
@click.option('--save_dir', type=str, default='models', help='Directory to save the trained model')
@click.option('--save_name', type=str, default='model', help='Name of the saved model')
@click.option('--logs', type=str, default='logs', help='Directory to save TensorBoard logs')
@click.option('--epoch', type=int, default=10, help='Number of epochs')
@click.option('--batch_size', type=int, default=4, help='Batch size')
@click.option('--lr', type=float, default=0.001, help='Learning rate')
@click.option('--patience', type=int, default=3, help='Patience for early stopping')
@click.option('--weight_decay', type=float, default=0.0, help='Weight decay')
@click.option('--image_size', type=str, default='512,512', help='Image size (width,height)')
@click.option('--train_dir', type=str, default='path/to/train', help='Path to the training data directory')
@click.option('--test_dir', type=str, default='path/to/test', help='Path to the testing data directory')
def train(save_dir, save_name, logs, epoch, batch_size, lr, patience,
          weight_decay, image_size, train_dir, test_dir):
    image_size = tuple(map(int, image_size.split(',')))

    train_image_dir = os.path.join(train_dir, 'images')
    train_mask_dir = os.path.join(train_dir, 'masks')
    test_image_dir = os.path.join(test_dir, 'images')
    test_mask_dir = os.path.join(test_dir, 'masks')

    # Create model save directory if it does not exist
    tf.io.gfile.makedirs(save_dir)

    # Split dataset into train and test sets
    train_image_filenames = glob.glob(os.path.join(train_image_dir, '*.jpg'))
    train_mask_filenames = glob.glob(os.path.join(train_mask_dir, '*.jpeg'))
    test_image_filenames = glob.glob(os.path.join(test_image_dir, '*.jpg'))
    test_mask_filenames = glob.glob(os.path.join(test_mask_dir, '*.jpeg'))

    click.secho("Number of train images: {}".format(len(train_image_filenames)), fg='green')
    click.secho("Number of train masks: {}".format(len(train_mask_filenames)), fg='green')
    click.secho("Number of test images: {}".format(len(test_image_filenames)), fg='green')
    click.secho("Number of test masks: {}".format(len(test_mask_filenames)), fg='green')

    # Create data loaders
    train_data_loader = DataGenerator(train_image_filenames, train_mask_filenames,
                                      batch_size, target_size=image_size, augmentations=True)
    test_data_loader = DataGenerator(test_image_filenames, test_mask_filenames,
                                     batch_size, target_size=image_size, augmentations=False)

    # Create and compile U-Net model
    input_shape = image_size + (3,)
    model = build_resnet50_unet(input_shape)
    optimizer = tfa.optimizers.AdamW(learning_rate=lr, weight_decay=weight_decay)
    model.compile(optimizer=optimizer, loss=weighted_bce_dice_loss,
                  metrics=[MeanIoU(num_classes=2), 'binary_accuracy', dice_coef])

    click.secho('Training started...', fg='green')

    # Define callbacks
    lr_reducer = ReduceLROnPlateau(factor=0.1,
                                   cooldown=1,
                                   patience=2, verbose=1,
                                   min_lr=0.1e-6)
    checkpoint = ModelCheckpoint(os.path.join(save_dir, save_name + '.h5'), monitor='val_loss', save_best_only=True)
    early_stopping = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
    tensorboard = TensorBoard(log_dir=logs)

    model.fit(train_data_loader,
        epochs=epoch,
        batch_size=1,
        validation_data=test_data_loader,
        callbacks=[checkpoint, early_stopping, tensorboard, lr_reducer]
    )

    click.secho('Training completed!', fg='green')

    # Save the final model
    model.save(os.path.join(save_dir, save_name + '_final.h5'))
    click.secho('Final model saved!', fg='green')


if __name__ == '__main__':
    train()
