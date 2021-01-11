"""
By: Jinseo Lee

Various ways to set up deep learning models in keras
"""

from matplotlib import pyplot as plt
import pickle

"""
Training model with 'fit_generator' in keras
Manually change parameters appropriately
"""
def train(model, training_set, validation_set, test_set):
    
    history = model.fit_generator(training_set, steps_per_epoch = 10, epochs = 8)
    history = model.fit_generator(trainGen, shuffle=True, steps_per_epoch=len(X_train) / bs_train, validation_data = valGen, validation_steps=len(X_val)/bs_val, epochs=20, callbacks=callbacks_list, verbose=2, max_queue_size=5)

    return history


def save_training_history(save_dir, history):
    # saving model training history as a dictionary
    with open(save_dir, 'wb') as file_pi:
        pickle.dump(history.history, file_pi)

def view_training_history(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.figure()
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()
    
    
    
    
    
