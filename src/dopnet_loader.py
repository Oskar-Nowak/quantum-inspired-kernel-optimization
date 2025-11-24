import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

class DopNetLoader:
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.data = {}
        self.labels = {}

    def _load_all(self):
        files = [f for f in os.listdir(self.root_dir) if f.endswith('.mat')]
        files.sort()

        print(f'Found {len(files)} .mat files')

        for fname in files:
            path = os.path.join(self.root_dir, fname)
            self._load_single_file(path)

        print('\n=== SUMMARY ===')
        for person, gestures in self.data.items():
            print(f"Person {person}: {len(gestures)} gestures × {len(gestures[0])} repetitions")

    def _load_single_file(self, path):
        d = loadmat(path, struct_as_record=False, squeeze_me=False)
        train = d['Data_Training'][0, 0]

        person = train.name[0]
        signals = train.Doppler_Signals[0]

        gestures = []

        for gesture_idx in range(4):
            gesture_cell = signals[gesture_idx]
            reps = [rep_obj[0] for rep_obj in gesture_cell]
            gestures.append(reps)

        self.data[person] = gestures

    # ---------------------------
    #  PUBLIC API
    # ---------------------------

    def get_raw_data(self):
        return self.data
    
    def to_feature_matrix(self, mode='magnitude'):
        X = list()
        y = list()

        for person, gestures in self.data.items():
            for gesture_idx, reps in enumerate(gestures):

                for rep in reps:
                    spec = rep

                    if np.iscomplexobj(spec):
                        if mode == 'magnitude':
                            spec = np.abs(spec)
                        elif mode == 'realimag':
                            spec = np.stack([spec.real, spec.imag], axis=-1)
                        else:
                            raise ValueError('Unknown mode')
                        
                    X.append(spec.flatten())
                    y.append(f'{person}_{gesture_idx}')

        X = np.array(X)
        y = np.array(y)
        return X, y
    
    def plot_spectrogram(self, person, gesture, rep, mode="magnitude"):
        spec = self.data[person][gesture][rep]

        if np.iscomplexobj(spec):
            if mode == "magnitude":
                spec = np.abs(spec)
            elif mode == "real":
                spec = spec.real
            elif mode == "imag":
                spec = spec.imag

        plt.figure(figsize=(10,4))
        plt.imshow(spec, aspect='auto', cmap='viridis')
        plt.title(f"Person {person} | Gesture {gesture} | Rep {rep}")
        plt.colorbar()
        plt.show()
