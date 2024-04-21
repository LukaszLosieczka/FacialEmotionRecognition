from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef
import numpy as np
import data_preprocessing as dp
import tensorflow as tf
from keras import layers, models
import json
import pickle
import sys
from keras.models import Model
from keras.applications import VGG16
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from joblib import dump, load


MODELS_PATH = 'experiment3/models'
RESULTS_PATH = 'experiment3/results'

INPUT_SHAPE = (224, 224, 3)
NUM_CLASSES = 7
BATCH_SIZE = 32

SVM = 'svm'
DT = 'decision_tree'
DNN = 'dnn'


def extract_features(base_model, data):
    x = layers.GlobalAveragePooling2D()(base_model.output)
    model = Model(inputs=base_model.input, outputs=x)
    for layer in base_model.layers:
        layer.trainable = False
    features = model.predict(data, verbose=1)
    return features


def train_dnn(base_model, train_data, val_data, epochs):
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
    with open(f'{MODELS_PATH}/train_history_dnn.pkl', 'wb') as file:
        pickle.dump(history.history, file)
    model.save(f'{MODELS_PATH}/model_dnn.h5')


def train_model(train_data, val_data, epochs, classifier):
    base_model = VGG16(include_top=False, weights='imagenet', input_shape=INPUT_SHAPE)
    if classifier == DNN:
        train_dnn(base_model, train_data, val_data, epochs)
        return
    print("Extracting features...")
    train_features = extract_features(base_model, train_data)
    print("Finished extracting features")
    train_labels = train_data.classes
    if classifier == SVM:
        classifier = SVC()
    else:
        classifier = DecisionTreeClassifier()
    print("Training classifier...")
    classifier.fit(train_features, train_labels)
    print("Training finished")
    dump(classifier, f'{MODELS_PATH}/model_{classifier}.pkl')


def test_model(model, test_data, is_dnn=False):
    if not is_dnn:
        base_model = VGG16(include_top=False, weights='imagenet', input_shape=INPUT_SHAPE)
        features = extract_features(base_model, test_data)
        y_pred = np.argmax(features, axis=-1)
    else:
        y_pred = np.argmax(model.predict(test_data), axis=-1)
    y_true = np.argmax(test_data.labels, axis=-1)
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
        is_dnn = len(arguments) == 3 and arguments[2] == '--dnn'
        if not is_dnn:
            loaded_model = load(f'{model_name}.pkl')
        else:
            loaded_model = tf.keras.models.load_model(f'{MODELS_PATH}/{model_name}.h5')
        result = test_model(loaded_model, test_data, is_dnn=is_dnn)
        with open(f'{RESULTS_PATH}/test_results_{model_name}.json', 'w') as json_file:
            json.dump(result, json_file)
        return

    train_data = []
    if arguments[1].lower() == 'raw':
        train_data = dp.get_train_data_raw()
    elif arguments[1].lower() == 'balanced':
        train_data = dp.get_train_data_balanced()

    classifier = arguments[2].lower()
    epochs = int(arguments[3].lower())

    if arguments[0].lower() == '--train':
        print('TRAINING')
        train_model(train_data, val_data, epochs, classifier)


if __name__ == '__main__':
    args = sys.argv[1:]
    main(args)