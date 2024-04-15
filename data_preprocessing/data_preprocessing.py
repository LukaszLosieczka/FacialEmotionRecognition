import os
import seaborn as sns
import matplotlib.pyplot as plt
import random
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from keras.utils import load_img, img_to_array, array_to_img


TRAIN_DIR = '../dataset/train'
TEST_DIR = '../dataset/test'
BALANCED_DIR = 'preprocessed_data'
CLASSES_DIR = ['/angry', '/disgust', '/fear', '/happy', '/neutral', '/sad', '/surprise']
IMAGE_SIZE = (48, 48)
TARGET_IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

EXPAND_DATAGEN = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)


AUGMENT_DATAGEN = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1/255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

RESCALE_DATAGEN = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1/255
)


def balance_dataset(data_dir: str, target_class_count: int = 8000):
    new_data_dir = 'preprocessed_data'
    os.makedirs(new_data_dir)
    class_names = os.listdir(data_dir)

    for class_index, class_name in enumerate(class_names):
        os.makedirs(os.path.join(new_data_dir, class_name))
        class_dir = os.path.join(data_dir, class_name)
        images = os.listdir(class_dir)
        class_images = []
        images_to_add = target_class_count - len(images)
        for image_name in images:
            image_path = os.path.join(class_dir, image_name)
            image = load_img(image_path, color_mode='rgb', target_size=TARGET_IMAGE_SIZE)
            class_images.append(image)
        if images_to_add > 0:
            fill_array(class_images, images_to_add)
        if images_to_add < 0:
            class_images = class_images[:target_class_count]

        for i, image in enumerate(class_images, 1):
            image.save(f'{new_data_dir}/{class_name}/image{i}.jpg')


def fill_array(array, images_to_add):
    current_length = len(array)
    for i in range(images_to_add):
        image_to_copy = array[np.random.randint(0, current_length)]
        image_to_copy = img_to_array(image_to_copy)
        image_to_copy = np.expand_dims(image_to_copy, axis=0)
        augmented_image = EXPAND_DATAGEN.flow(image_to_copy, batch_size=1)
        array.append(array_to_img(next(augmented_image)[0]))
        current_length += 1


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


def get_train_data_balanced_augmented():
    return AUGMENT_DATAGEN.flow_from_directory(
        BALANCED_DIR,
        target_size=TARGET_IMAGE_SIZE,
        color_mode='rgb',
        class_mode='categorical',
        batch_size=BATCH_SIZE
    )


def get_train_data_augmented():
    return AUGMENT_DATAGEN.flow_from_directory(
        TRAIN_DIR,
        target_size=TARGET_IMAGE_SIZE,
        color_mode='rgb',
        class_mode='categorical',
        batch_size=BATCH_SIZE
    )


def get_train_data_raw():
    return RESCALE_DATAGEN.flow_from_directory(
        TRAIN_DIR,
        target_size=TARGET_IMAGE_SIZE,
        color_mode='rgb',
        class_mode='categorical',
        batch_size=BATCH_SIZE
    )


def get_validation_and_test_data():
    valid_df, test_df = split_test_data(0.5)
    return RESCALE_DATAGEN.flow_from_dataframe(
        valid_df,
        target_size=TARGET_IMAGE_SIZE,
        color_mode='rgb',
        class_mode='categorical',
        batch_size=BATCH_SIZE
    ), RESCALE_DATAGEN.flow_from_dataframe(
        test_df,
        target_size=TARGET_IMAGE_SIZE,
        color_mode='rgb',
        class_mode='categorical',
        batch_size=BATCH_SIZE
    )


def split_test_data(valid_size):
    filepaths = []
    labels = []

    folds = os.listdir(TEST_DIR)
    for fold in folds:
        foldpath = os.path.join(TEST_DIR, fold)
        filelist = os.listdir(foldpath)
        for file in filelist:
            fpath = os.path.join(foldpath, file)
            filepaths.append(fpath)
            labels.append(fold)

    # Concatenate data paths with labels into one dataframe
    Fseries = pd.Series(filepaths, name='filepaths')
    Lseries = pd.Series(labels, name='labels')
    test_dataframe = pd.concat([Fseries, Lseries], axis=1)

    return train_test_split(test_dataframe, train_size=valid_size, shuffle=True, random_state=123)


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
    # show_classes_counts(PREPROCESSED_DIR)
    show_classes_samples(BALANCED_DIR)
    # get_train_data_generator()
    # balance_dataset(TEST_DIR, target_class_count=1000)
