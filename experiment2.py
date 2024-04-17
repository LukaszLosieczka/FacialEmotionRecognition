import json
import pickle
import sys
import cv2
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef
import numpy as np
import data_preprocessing as dp
import tensorflow as tf
from keras import layers, models

from keras.applications import ResNet50V2, EfficientNetB0, VGG16

MODELS_PATH = 'experiment2/models'
RESULTS_PATH = 'experiment2/results'

RESNET = 'resnet'
VGG = 'vgg16'
EFFNET = 'effnet'
HOG = 'hog'

INPUT_SHAPE = (224, 224, 3)
NUM_CLASSES = 7
BATCH_SIZE = 32


def transform_data_from_datagen(datagen):
    hog_features = []
    labels = []
    for images_batch, labels_batch in datagen:
        for image in images_batch:
            resized_image = cv2.resize(image, (64, 64))
            gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
            image_uint8 = cv2.convertScaleAbs(gray_image)
            hog_feature = cv2.HOGDescriptor((64, 64), (16, 16), (8, 8), (8, 8), 9).compute(image_uint8)
            hog_features.append(hog_feature.flatten())
        labels.extend(labels_batch)
        if len(hog_features) >= datagen.samples:
            break

    X = np.array(hog_features)
    y = np.array(labels)

    return X, y


def train_model_with_hog(train_data, val_data, epochs):
    print("Extracting features using hog...")
    train_features, train_labels = transform_data_from_datagen(train_data)
    val_features, val_labels = transform_data_from_datagen(val_data)
    print("Features extracted!")
    model = models.Sequential([
        layers.Dense(256, activation='relu', input_shape=(train_features.shape[1],)),
        layers.Dense(128, activation='relu'),
        layers.Dense(NUM_CLASSES, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(train_features, train_labels, epochs=epochs,
                        batch_size=32, validation_data=(val_features, val_labels))
    with open(f'{MODELS_PATH}/train_history_hog.pkl', 'wb') as file:
        pickle.dump(history.history, file)
    return model


def train_model(train_data, val_data, epochs, feature_extraction):
    if feature_extraction == HOG:
        return train_model_with_hog(train_data, val_data, epochs)

    if feature_extraction == RESNET:
        base_model = ResNet50V2(include_top=False, weights='imagenet', input_shape=INPUT_SHAPE)
    elif feature_extraction == VGG:
        base_model = VGG16(include_top=False, weights='imagenet', input_shape=INPUT_SHAPE)
    else:
        base_model = EfficientNetB0(include_top=False, weights='imagenet', input_shape=INPUT_SHAPE)

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(NUM_CLASSES, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    history = model.fit(train_data, epochs=epochs, validation_data=val_data)
    with open(f'{MODELS_PATH}/train_history_{feature_extraction}.pkl', 'wb') as file:
        pickle.dump(history.history, file)
    return model


def test_model(model, test_data, is_hog=False):
    if is_hog:
        X_test, y_test = transform_data_from_datagen(test_data)
        y_pred = np.argmax(model.predict(X_test), axis=-1)
        y_true = np.argmax(y_test, axis=-1)
    else:
        y_pred = np.argmax(model.predict(test_data), axis=-1)  # Predict classes for test data
        y_true = np.argmax(test_data.labels, axis=-1)  # True labels for test data
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    mcc = matthews_corrcoef(y_true, y_pred)
    result = {
        "test_accuracy": float(accuracy),
        "test_recall": float(recall),
        "test_precision": float(precision),
        "test_f1": float(f1),
        "test_mcc": float(mcc)
    }
    print(result)
    return result


def main(arguments):
    print('Loading validation and test data')
    val_data, test_data = dp.get_validation_and_test_data()

    if arguments[0].lower() == '--test':
        print('TESTING')
        model_name = arguments[1]
        is_hog = len(arguments) == 3 and arguments[2] == '--hog'
        loaded_model = tf.keras.models.load_model(f'{MODELS_PATH}/{model_name}.h5')
        result = test_model(loaded_model, test_data, is_hog=is_hog)
        with open(f'{RESULTS_PATH}/test_results_{model_name}.json', 'w') as json_file:
            json.dump(result, json_file)
        return

    train_data = []
    if arguments[1].lower() == 'raw':
        train_data = dp.get_train_data_raw()
    elif arguments[1].lower() == 'balanced':
        train_data = dp.get_train_data_balanced()

    feature_extraction = arguments[2].lower()
    epochs = int(arguments[3].lower())

    if arguments[0].lower() == '--train':
        print('TRAINING')
        model = train_model(train_data, val_data, epochs, feature_extraction)
        model.save(f'{MODELS_PATH}/model_{arguments[2].lower()}.h5')


if __name__ == '__main__':
    args = sys.argv[1:]
    main(args)
