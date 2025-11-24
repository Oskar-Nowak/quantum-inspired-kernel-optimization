import os
from scipy.io import loadmat

class DopNetLoader:
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.data = {}
        self.labels = {}

    def load_all(self):
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
        d = loadmat(path, struct_as_record = False, squeeze_me = False)
        train = d['Data_Training'][0, 0]

        # Person label (A-F)
        name = train.name[0]

        # Gestures stored as 1x4 MATLAB cell -> NumPy array shape (1,4)
        signals = train.Doppler_Signals[0]

        # Final list: [gesture0_reps, gesture1_reps, gesture2_reps, gesture3_reps]
        gestures = list() 

        for gesture_idx in range(4):
            gesture_cell = signals[gesture_idx]
            reps = list()

            for rep_obj in gesture_cell:
                spec = rep_obj[0]
                reps.append(spec)

            gestures.append(reps)

        self.data[name] = gestures
        self.labels[name] = [(name, g) for g in range(4) for _ in range(len(gestures[0]))]