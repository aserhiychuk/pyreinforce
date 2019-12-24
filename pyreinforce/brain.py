

class Brain(object):
    """Hold neural network(s) implementation.

    The purpose of this class is to encapsulate neural network(s)
    implementation from the agent. Agents typically use it to
    obtain/predict possible actions, and then train the network.

    Please refer to the `/examples` folder for some possible
    usages of the `Brain` class.
    """

    def train(self, *args, **kwargs):
        """Perform a training step.

        Note
        ----
        Subclasses must implement this method.
        """
        raise NotImplementedError

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass
