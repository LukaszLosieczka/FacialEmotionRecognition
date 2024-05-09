import time

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef
import numpy as np

import data_preprocessing as dp
import tensorflow as tf
from keras import layers, models
import json
import pickle
import sys
from keras.models import Model
from keras.applications import ResNet50V2, MobileNet
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from joblib import dump, load


MODELS_PATH = '/content/drive/MyDrive/colab_env/FacialEmotionRecognition/experiment3/models'
RESULTS_PATH = '/content/drive/MyDrive/colab_env/FacialEmotionRecognition/experiment3/results'

INPUT_SHAPE = (224, 224, 3)
NUM_CLASSES = 7
BATCH_SIZE = 32

SVM_LINEAR = 'svm-linear'
SVM = 'svm'
DT = 'decision_tree'
RF = 'random_forest'
DNN = 'dnn'


def extract_features(base_model, data):
    x = layers.GlobalAveragePooling2D()(base_model.output)
    model = Model(inputs=base_model.input, outputs=x)
    for layer in base_model.layers:
        layer.trainable = False
    features = model.predict(data, verbose=1)
    return features


def train_dnn(base_model, train_data, val_data, epochs):
    for layer in base_model.layers[:-4]:
        layer.trainable = False
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.1),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.1),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.1),
        layers.Dense(NUM_CLASSES, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    history = model.fit(train_data, epochs=epochs, validation_data=val_data)
    with open(f'{MODELS_PATH}/train_history_dnn_{epochs}.pkl', 'wb') as file:
        pickle.dump(history.history, file)
    model.save(f'{MODELS_PATH}/model_dnn_{epochs}.h5')
    return model


def train_model(train_data, val_data, epochs, classifier_name):
    base_model = ResNet50V2(include_top=False, weights='imagenet', input_shape=INPUT_SHAPE)
    if classifier_name == DNN:
        return train_dnn(base_model, train_data, val_data, epochs)
    print("Extracting features...")
    train_features = extract_features(base_model, train_data)
    print("Finished extracting features")
    train_labels = train_data.classes
    if classifier_name == SVM:
        classifier = SVC()
    elif classifier_name == SVM_LINEAR:
        classifier = SVC(kernel='linear')
    elif classifier_name == RF:
        classifier = RandomForestClassifier()
    else:
        classifier = DecisionTreeClassifier()
    print("Training classifier...")
    classifier.fit(train_features, train_labels)
    print("Training finished")
    dump(classifier, f'{MODELS_PATH}/model_{classifier_name}.pkl')
    return classifier


def test_model(model, test_data, is_dnn=False):
    if not is_dnn:
        base_model = ResNet50V2(include_top=False, weights='imagenet', input_shape=INPUT_SHAPE)
        features = extract_features(base_model, test_data)
        y_pred = model.predict(features)
    else:
        y_pred = np.argmax(model.predict(test_data), axis=-1)
    y_true = test_data.classes
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
    val_data, test_data = dp.get_validation_and_test_data_mobilenet()

    if arguments[0].lower() == '--test':
        print('TESTING')
        model_name = arguments[1]
        is_dnn = len(arguments) == 3 and arguments[2] == '--dnn'
        if not is_dnn:
            loaded_model = load(f'{MODELS_PATH}/{model_name}.pkl')
        else:
            loaded_model = tf.keras.models.load_model(f'{MODELS_PATH}/{model_name}.h5')
        result = test_model(loaded_model, test_data, is_dnn=is_dnn)
        with open(f'{RESULTS_PATH}/test_results_{model_name}.json', 'w') as json_file:
            json.dump(result, json_file)
        return

    train_data = []
    if arguments[1].lower() == 'raw':
        train_data = dp.get_train_data_raw_mobilenet()
    elif arguments[1].lower() == 'balanced':
        train_data = dp.get_train_data_balanced_mobilenet()

    classifier_name = arguments[2].lower()
    epochs = int(arguments[3].lower())

    if arguments[0].lower() == '--train':
        print('TRAINING')
        start_time = time.time()
        model = train_model(train_data, val_data, epochs, classifier_name)
        end_time = time.time()
        print(f"Training time: {end_time - start_time} seconds")
        test_model(model, test_data, is_dnn=classifier_name == DNN)


if __name__ == '__main__':
    args = sys.argv[1:]
    main(args)
