import pickle

from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns


def conf_matrix(true_labels, pred_labels, classes):
    cm = confusion_matrix(true_labels, pred_labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="magma", xticklabels=classes, yticklabels=classes)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title('Confusion Matrix')
    plt.show()


def analyze_train_history(train_history):
    plt.figure(figsize=(15, 10))

    plt.subplot(2, 1, 1)
    plt.plot(train_history['loss'])
    plt.plot(train_history['val_loss'], 'ro')
    plt.title('Loss')
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.plot(train_history['accuracy'])
    plt.plot(train_history['val_accuracy'], 'ro')
    plt.title('Accuracy')
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    with open('experiment1/models/train_history_balanced5.pkl', 'rb') as file:
        train_hist = pickle.load(file)
        analyze_train_history(train_hist)
