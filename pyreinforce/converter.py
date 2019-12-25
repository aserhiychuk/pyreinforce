

class Converter(object):
    """Base class for all converters."""

    def convert_state(self, s, info=None):
        """Convert state to be consumable by the neural network.

        Base implementation returns original state.

        Parameters
        ----------
        s : obj
            State as it is received from the environment.

        Returns
        -------
        obj
            State that can be fed into the neural network.
        """
        return s

    def convert_action(self, a):
        """Convert action to be consumable by the environment.

        Base implementation returns original action.

        Parameters
        ----------
        a : obj
            Action as it is received from the neural network.

        Returns
        -------
        obj
            Action that can be fed into the environment.
        """
        return a

    def convert_experience(self, experience, info=None):
        return experience
