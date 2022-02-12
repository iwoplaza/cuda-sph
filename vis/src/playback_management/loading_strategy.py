

class LoadingStrategy:
    """
    Responsible for providing data at a random time, from a continuous and finite dataset.
    """

    def get_data_at_time(self, time: float):
        """
        :param: time in seconds
        """
        raise NotImplementedError()

    def get_duration(self):
        raise NotImplementedError()
