from .loading_strategy import LoadingStrategy


PAUSED = 0
PLAYING = 1


class PlaybackManager:
    """
    Responsible for handling playback state for a continuous and finite dataset.
    """

    def __init__(self, loading_strategy: LoadingStrategy):
        self.state = PAUSED
        self.time_elapsed = 0
        self.play_rate = 1
        self.__looping = False

        self.loading_strategy = loading_strategy

    def get_current_data(self):
        return self.loading_strategy.get_data_at_time(self.time_elapsed)

    def get_time_elapsed(self):
        return self.time_elapsed

    def get_playback_duration(self):
        return self.loading_strategy.get_duration()

    def is_looping(self):
        return self.__looping

    def update(self, delta_time: float) -> None:
        if self.state == PLAYING:
            self.time_elapsed += delta_time * self.play_rate

            dur = self.get_playback_duration()
            if self.time_elapsed > dur:
                if self.__looping:
                    self.time_elapsed = self.time_elapsed % dur
                else:
                    self.time_elapsed = dur
                    self.state = PAUSED

    def set_state(self, state: int) -> None:
        self.state = state

    def get_state(self) -> int:
        return self.state

    def set_looping(self, looping):
        self.__looping = looping

    def play(self) -> None:
        self.set_state(PLAYING)

    def pause(self) -> None:
        self.set_state(PAUSED)

    def set_play_rate(self, rate) -> None:
        self.play_rate = rate
