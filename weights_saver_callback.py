from keras.callbacks import Callback


class WeightsSaverCallback(Callback):
    def __init__(self, model, every):
        self.model = model
        self.N = every
        self.batch = 0

    def on_batch_end(self, batch, logs={}):
        if self.batch % self.N == 0:
            name = 'weights/weights%08d.h5' % self.batch
            self.model.save_weights(name)
        self.batch += 1
