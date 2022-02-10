from typing import Tuple
from common.data_classes import Pipe, Segment


class PipeBuilder:
    """
    Pipe builder makes creating Pipe object much easier.
    Firstly specify your starting segment, then sequentially add another - they will adapt to previous to each other
    Notice that segments are created in way the methods are called and position and radius of first segment need to be
    set before adding new segments.
    """
    _FIRST_SEGMENT_MESSAGE = "First segment can't be change after adding new segments"
    _NEGATIVE_RADIUS_MESSAGE = "Radius must be positive"
    _NEGATIVE_CHANGE_MESSAGE = "Change must be positive"
    _NEGATIVE_LENGTH_MESSAGE = "Length must be positive"

    def __init__(self) -> None:
        self.__first_segment = True
        self.__pipe: Pipe = Pipe(segments=[Segment()])

    def with_starting_position(self, position: Tuple[float, float, float]) -> 'PipeBuilder':
        """
        Sets centre point of first segment

        :param position: Centre starting point of first segment
        :return: Builder object
        """
        assert self.__first_segment, self._FIRST_SEGMENT_MESSAGE
        self.__pipe.segments[0].start_point = position
        return self

    def with_starting_radius(self, radius: float) -> 'PipeBuilder':
        """
        Sets radius at start of first segment

        :param radius: Radius of first segment
        :return: Builder object
        """
        assert self.__first_segment, self._FIRST_SEGMENT_MESSAGE
        assert radius > 0, self._NEGATIVE_RADIUS_MESSAGE
        self.__pipe.segments[0].start_radius = radius
        return self

    def with_ending_radius(self, radius: float) -> 'PipeBuilder':
        """
        Sets radius at ends of first segment

        :param radius: Radius at ends of first segment
        :return: Builder object
        """
        assert self.__first_segment, self._FIRST_SEGMENT_MESSAGE
        assert radius > 0, self._NEGATIVE_RADIUS_MESSAGE
        self.__pipe.segments[0].end_radius = radius
        return self

    def with_starting_length(self, length: float) -> 'PipeBuilder':
        """
        Sets length of first segment

        :param length: Length of first segment
        :return: Builder object
        """
        assert self.__first_segment, self._FIRST_SEGMENT_MESSAGE
        assert length > 0, self._NEGATIVE_LENGTH_MESSAGE
        self.__pipe.segments[0].length = length
        return self

    def add_roller_segment(self, length) -> 'PipeBuilder':
        """
        Adds segment of specified length with

        :param length:
        :return: Builder object
        """
        self.__first_segment = False
        assert length > 0, self._NEGATIVE_LENGTH_MESSAGE

        last_segment = self.__pipe.segments[-1]
        new_start_point = (last_segment.start_point[0]+last_segment.length, last_segment.start_point[1],
                           last_segment.start_point[2])

        self.__pipe.segments.append(Segment(
            start_point=new_start_point,
            start_radius=last_segment.end_radius,
            end_radius=last_segment.end_radius,
            length=length))

        return self

    def add_lessening_segment(self, length, change) -> 'PipeBuilder':
        """
        Adds a segment new segment which is a truncated cone with radius at end smaller than at start by a change
        factor

        :param length: Length of a new segment
        :param change: Change in radius at the end of the segment compare to the beginning
        :return: Builder object
        """
        self.add_roller_segment(length)
        assert change > 0, self._NEGATIVE_CHANGE_MESSAGE
        assert change < self.__pipe.segments[-1].end_radius, "After change radius must be positive"
        self.__pipe.segments[-1].end_radius = self.__pipe.segments[-1].end_radius - change
        return self

    def add_increasing_segment(self, length, change) -> 'PipeBuilder':
        """
        Adds a new segment which is a truncated cone with radius at the end bigger than at the beginning by a change
        factor

        :param length: Length of a new segment
        :param change: Change in radius at the end of the segment compare to the beginning
        :return: Builder object
        """
        self.add_roller_segment(length)
        assert change > 0, self._NEGATIVE_CHANGE_MESSAGE
        self.__pipe.segments[-1].end_radius = self.__pipe.segments[-1].end_radius + change
        return self

    def get_result(self) -> Pipe:
        """
        :return: Built Pipe object
        """
        return self.__pipe
