import json
import pickle

from keras import layers, models
import numpy as np
import sys
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef
import data_preprocessing as dp
import tensorflow as tf
from keras.applications import ResNet50V2

MODELS_PATH = 'experiment1/models'
RESULTS_PATH = 'experiment1/results'

INPUT_SHAPE = (224, 224, 3)
NUM_CLASSES = 7
EPOCHS = 10


def create_cnn_model(input_shape, num_classes):
    model = models.Sequential([
        # Convolutional layers
        # Conv 1
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        # Conv 2
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        # Conv 3
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        # Conv 4
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dropout(0.5),

        # Dense layers
        layers.Dense(512, activation='relu'),
        layers.Dense(256, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model


def create_pretrained_model(input_shape, num_classes):
    base_model = ResNet50V2(include_top=False, weights='imagenet', input_shape=input_shape)
    model = models.Sequential([
        base_model,
        layers.BatchNormalization(),
        layers.GlobalAveragePooling2D(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.1),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.1),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.1),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model


def train_model(train_data, val_data, use_class_weight=False):
    class_weight = dp.get_classes_weights(train_data)
    model = create_pretrained_model(INPUT_SHAPE, NUM_CLASSES) #create_cnn_model(INPUT_SHAPE, NUM_CLASSES)
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    history = model.fit(train_data, epochs=EPOCHS, validation_data=val_data,
                        class_weight=class_weight if use_class_weight else None)
    with open(f'{MODELS_PATH}/train_history.pkl', 'wb') as file:
        pickle.dump(history.history, file)
    return model


def test_model(model, test_data, save_result=True):
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
    if save_result:
        with open(f'{RESULTS_PATH}/test_results.json', 'w') as json_file:
            json.dump(result, json_file)


def main(arguments):
    train_data = []
    print('Loading validation and test data')
    val_data, test_data = dp.get_validation_and_test_data()

    if arguments[0].lower() == '--test':
        print('TESTING')
        loaded_model = tf.keras.models.load_model(f'{MODELS_PATH}/model.h5')
        test_model(loaded_model, test_data, save_result=True)
        return

    use_weights = False
    if arguments[1].lower() == 'raw':
        print("Using raw training dataset")
        train_data = dp.get_train_data_raw()
    elif arguments[1].lower() == 'balanced':
        print("Using balanced training dataset")
        train_data = dp.get_train_data_balanced()
    elif arguments[1].lower() == 'augmented':
        print("Using augmented training dataset")
        train_data = dp.get_train_data_augmented()
    elif arguments[1].lower() == 'augmented+balanced':
        print("Using balanced and augmented training dataset")
        train_data = dp.get_train_data_balanced_augmented()

    if arguments[0].lower() == '--train':
        print('TRAINING')
        model = train_model(train_data, val_data, use_class_weight=use_weights)
        model.save(f'{MODELS_PATH}/model_{arguments[1].lower()}.h5')


if __name__ == '__main__':
    args = sys.argv[1:]
    main(args)
