# This file runs all the unit tests implemented
from sa_utils import load_data as load_imdb
from lm_utils import load_data as load_lm_data
from mm_utils import load_data as load_mm_data
import time


def unit_test_1_through_9(unit_test):
    """
        This function unit tests load times (they must load in less than 5 minutes) for each data set, an

        Input:
            unit_test: int, unit test index to run.
    """

    start_time = time.time()

    if unit_test == 1:
        _, _, _, _, _ = load_imdb(num_words=24902, trim_length=2697)
    elif unit_test == 2:
        _, _, _, _ = load_lm_data("enwik8")
    elif unit_test == 3:
        _, _, _, _ = load_lm_data("text8")
    elif unit_test == 4:
        _, _, _, _ = load_lm_data("penn")
    elif unit_test == 5:
        _, _, _, _ = load_lm_data("pennchar")
    elif unit_test == 6:
        _, _, _, _ = load_mm_data("JSB Chorales")
    elif unit_test == 7:
        _, _, _, _ = load_mm_data("Nottingham")
    elif unit_test == 8:
        _, _, _, _ = load_mm_data("MuseData")
    elif unit_test == 9:
        _, _, _, _ = load_mm_data("Piano-midi.de")

    end_time = time.time()

    time_taken = end_time - start_time

    load_speed_in_seconds = 5 * 60  # 5 minutes with 60 seconds each

    if time_taken < load_speed_in_seconds:
        print(f"Unit test {unit_test} passed! Data set loaded in {time_taken} seconds!")
    else:
        print(f"Unit test {unit_test} failed! Data set loaded in {time_taken} seconds!")


if __name__ == '__main__':  # Main function
    for i in range(1, 10):
        unit_test_1_through_9(i)
