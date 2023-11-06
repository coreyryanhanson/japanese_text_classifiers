import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score


def determine_trainable_layers(model):
    for layer in model.layers:
        print(layer.name, layer.trainable)
    print(f"{len(model.trainable_weights)} trainable weights")


def generate_keras_model(model, layers=[], compile_kwargs={}):
    model = model
    for layer in layers:
        model.add(layer)
    model.compile(**compile_kwargs)
    model.summary()
    return model

def image_class_accuracy_scores(y_act, y_hat):
    acc = accuracy_score(y_act, y_hat)
    bal_acc = balanced_accuracy_score(y_act, y_hat)
    print(f"Accuracy: {acc}")
    print(f"Balanced Accuracy: {bal_acc}")
    return acc, bal_acc

def image_class_evaluation(model, X, y):
    y_act = np.argmax(y, axis = 1)
    y_hat = np.argmax(model.predict(X), axis=1)
    return image_class_accuracy_scores(y_act, y_hat)