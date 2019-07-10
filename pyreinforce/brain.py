import tensorflow as tf


class Brain(object):
    '''
    TODO Brain class
    '''
    def __init__(self, sess=None):
        super().__init__()

        self._sess = sess

    def train(self, batch):
        raise NotImplementedError('Subclasses must implement {}#train() method'.format(type(self).__name__))

    def __enter__(self):
        if not self._sess:
            self._sess = tf.Session()

        self._sess.__enter__()

        init = tf.global_variables_initializer()
        self._sess.run(init)

        return self

    def __exit__(self, *args):
        self._sess.__exit__(*args)
