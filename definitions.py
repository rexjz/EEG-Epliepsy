import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_ROOT = os.path.join(ROOT_DIR, "data")

EpilepticEEGDataset = os.path.join(DATA_ROOT, "EpilepticEEGDataset")

EpilepticEEGDataset_segments = os.path.join(EpilepticEEGDataset, 'segments')
EpilepticEEGDataset_preprocessed_segments = os.path.join(EpilepticEEGDataset, 'preprocessed_segments')
EpilepticEEGDataset_pieces = os.path.join(EpilepticEEGDataset, 'pieces')
EpilepticEEGDataset_artifact_free_segments = os.path.join(EpilepticEEGDataset, 'artifact_free_segments')