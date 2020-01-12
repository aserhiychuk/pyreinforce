

class Converter(object):
    """Base class for all converters."""

    def convert_state(self, s, info=None):
        """Convert state to be consumable by the neural network.

        Base implementation returns original state.

        Parameters
        ----------
        s : obj
            State as it is received from the environment.
        info : dict or obj, optional
            Diagnostic information.

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
        """Convert experience to be stored in the replay memory.

        Base implementation returns original experience.

        Parameters
        ----------
        experience : sequence or obj
            Experience, typically a tuple of (`s`, `a`, `r`, `s1`,
            `terminal_flag`).
        info : dict or obj, optional
            Diagnostic information.

        Returns
        -------
        sequence or obj
            Experience that can be stored in the replay memory.
        """
        return experience
