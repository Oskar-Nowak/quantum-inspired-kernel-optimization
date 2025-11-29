import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.ndimage import zoom

GESTURE_NAMES = ["Wave", "Pinch", "Swipe", "Click"]


class DopNetLoader:
    def __init__(self, root_dir: str):
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
            print(f'Person: {person}:')
            for i, reps in enumerate(gestures):
                print(f'  Gesture {i}: {len(reps)} samples')

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

    def _normalize_shape(self, target_shape: tuple = (256, 256)):
        new_data = {}

        for person, gestures in self.data.items():
            new_gestures = []
            for reps in gestures:
                new_reps = []
                for spec in reps:
                    spec_mag = np.abs(spec)

                    zoom_h = target_shape[0] / spec_mag.shape[0]
                    zoom_w = target_shape[1] / spec_mag.shape[1]

                    spec_norm = zoom(spec_mag, (zoom_h, zoom_w))
                    new_reps.append(spec_norm)

                new_gestures.append(new_reps)
            new_data[person] = new_gestures

        self.data = new_data

    # ---------------------------
    #  PUBLIC API
    # ---------------------------

    def load_normalized(self, target_shape: tuple = (256, 256)) -> dict:
        """
        Loads the Dop-Net dataset, and normalize the data to the shame shape.
        """
        self._load_all()
        self._normalize_shape(target_shape)
        return self.data

    def to_feature_matrix(self, mode: str = 'magnitude'):
        X = list()
        y = list()

        for person, gestures in self.data.items():
            for gesture_idx, reps in enumerate(gestures):

                gesture_name = GESTURE_NAMES[gesture_idx]

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
                    y.append(gesture_name)

        X = np.array(X)
        y = np.array(y)
        return X, y
    
    def plot_spectrogram(self, person, gesture, rep, mode: str = 'magnitude'):
        spec = self.data[person][gesture][rep]

        if np.iscomplexobj(spec):
            if mode == 'magnitude':
                spec = np.abs(spec)
            elif mode == 'real':
                spec = spec.real
            elif mode == 'imag':
                spec = spec.imag

        plt.figure(figsize=(10,4))
        plt.imshow(spec, aspect='auto', cmap='viridis')
        plt.title(f'Person {person} | Gesture {gesture} | Rep {rep}')
        plt.colorbar()
        plt.show()
