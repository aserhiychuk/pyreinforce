import tensorflow as tf


class Brain(object):
    '''
    TODO Brain class
    '''
    def __init__(self):
        super().__init__()

        tf.reset_default_graph()
        self._sess = None

    def train(self, batch):
        raise NotImplementedError('Subclasses must implement {}#train() method'.format(type(self).__name__))

    def __enter__(self):
        self._sess = tf.Session()
        self._sess.__enter__()

        # initialize new model
        init = tf.global_variables_initializer()
        self._sess.run(init)

        return self

    def __exit__(self, *args):
        self._sess.__exit__(*args)
