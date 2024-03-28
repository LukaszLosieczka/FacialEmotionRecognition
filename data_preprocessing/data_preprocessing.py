import os
import seaborn as sns
import matplotlib.pyplot as plt
import random
import tensorflow as tf
import numpy as np
from sklearn.utils.class_weight import compute_class_weight


TRAIN_DIR = '../dataset/train'
TEST_DIR = '../dataset/test'
IMAGE_SIZE = (48, 48)
TARGET_IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

TRAIN_DATAGEN = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1/255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

TEST_DATAGEN = test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1/255
)


def show_classes_counts(data_dir: str):
    emotions = os.listdir(TRAIN_DIR)
    samples_count = []
    for emotion in emotions:
        folder = os.path.join(data_dir, emotion)
        samples_count.append(len(os.listdir(folder)))
    sns.barplot(x=emotions, y=samples_count, palette='inferno')
    plt.show()


def show_classes_samples(data_dir: str):
    plt.figure(figsize=(15, 10))
    emotions = os.listdir(TRAIN_DIR)

    for i, emotion in enumerate(emotions, 1):
        folder = os.path.join(data_dir, emotion)
        folder_size = len(os.listdir(folder))
        img_path = os.path.join(folder, os.listdir(folder)[random.randint(0, folder_size-1)])
        img = plt.imread(img_path)
        plt.subplot(2, 4, i)
        plt.imshow(img, cmap='gray')
        plt.title(emotion)
        plt.axis('off')
    plt.show()


def get_train_data_preprocessed():
    return TRAIN_DATAGEN.flow_from_directory(
        TRAIN_DIR,
        target_size=TARGET_IMAGE_SIZE,
        color_mode='rgb',
        class_mode='categorical',
        batch_size=BATCH_SIZE
    )


def get_test_data_preprocessed():
    return TEST_DATAGEN.flow_from_directory(
        TEST_DIR,
        target_size=TARGET_IMAGE_SIZE,
        color_mode='rgb',
        class_mode='categorical',
        batch_size=BATCH_SIZE
    )


def get_classes_weights(data):
    classes = np.array(data.classes)

    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(classes),
        y=classes
    )

    class_weights_dict = dict(enumerate(class_weights))
    return class_weights_dict


def show_preprocessed_data(data):
    images, labels = next(data)
    classes = list(data.class_indices.keys())
    plt.figure(figsize=(10, 10))
    for i in range(4):
        plt.subplot(2, 2, i + 1)
        plt.imshow(images[i])
        plt.title(classes[np.argmax(labels[i])])
        plt.axis('off')
    plt.show()


if __name__ == '__main__':
    print(tf.config.list_physical_devices('GPU'))
    # show_classes_counts(TRAIN_DIR)
    show_classes_samples(TRAIN_DIR)
    # get_train_data_generator()
    show_preprocessed_data(get_train_data_preprocessed())
