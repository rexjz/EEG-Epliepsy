import sys
from definitions import EpilepticEEGDataset
input_p_code = "p10"
if len(sys.argv) == 2:
    input_p_code = sys.argv[1]
    print("process " + sys.argv[1])


def get_record_number():
    None

def get_segment_abs_path():
    None


def artifact_removal():
    None


def process(p_code):
    None


process(input_p_code)
