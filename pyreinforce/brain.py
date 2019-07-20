

class Brain(object):
    '''
    TODO Brain class
    '''
    def train(self, *args, **kwargs):
        raise NotImplementedError

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass
