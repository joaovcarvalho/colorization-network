from keras.callbacks import Callback


class WeightsSaverCallback(Callback):
    def __init__(self, model, every):
        super(WeightsSaverCallback, self).__init__()
        self.model = model
        self.N = every
        self.batch = 0

    def on_batch_end(self, batch, logs=None):
        if self.batch % self.N == 0:
            name = 'weights/weights.h5'
            self.model.save_weights(name)
        self.batch += 1
