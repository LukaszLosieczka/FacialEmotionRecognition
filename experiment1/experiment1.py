import json
import pickle

from keras import layers, models
import numpy as np
import sys
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef
from data_preprocessing import data_preprocessing as dp
import tensorflow as tf

MODELS_PATH = 'models'
RESULTS_PATH = 'results'

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


def train_model(train_data, val_data, use_class_weight=False, save_model=True):
    class_weight = dp.get_classes_weights(train_data)
    model = create_cnn_model(INPUT_SHAPE, NUM_CLASSES)
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    history = model.fit(train_data, epochs=EPOCHS, validation_data=val_data,
                        class_weight=class_weight if use_class_weight else None)
    with open(f'{MODELS_PATH}/train_history.pkl', 'wb') as file:
        pickle.dump(history.history, file)
    if save_model:
        model.save(f'{MODELS_PATH}/model.h5')


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
    elif arguments[1].lower() == 'raw+weight':
        print("Using raw training dataset with classes weight")
        train_data = dp.get_train_data_raw()
        use_weights = True
    elif arguments[1].lower() == 'augmented':
        print("Using augmented training dataset with classes weight")
        train_data = dp.get_train_data_augmented()
        use_weights = True
    elif arguments[1].lower() == 'augmented+balanced':
        print("Using balanced and augmented training dataset")
        train_data = dp.get_train_data_balanced_augmented()

    if arguments[0].lower() == '--train':
        print('TRAINING')
        train_model(train_data, val_data, use_class_weight=use_weights, save_model=True)


if __name__ == '__main__':
    args = sys.argv[1:]
    main(args)
