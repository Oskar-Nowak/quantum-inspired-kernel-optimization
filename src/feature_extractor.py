import numpy as np
from enums import GESTURE_NAMES, ExtractorMode


class FeatureExtractor:
    def __init__(self, mode: ExtractorMode = ExtractorMode.Magnitude):
        self.mode = mode

    def transform(self, data_dict: dict):
        X = list()
        y = list()

        print("=== FeatureExtractor ===")
        print(f"Mode: {self.mode}")

        for person, gestures in data_dict.items():
            print(f"  Processing person: {person}")
            for gesture_idx, reps in enumerate(gestures):
                gesture_name = GESTURE_NAMES[gesture_idx]
                print(f"    Gesture: {gesture_name}, repetitions: {len(reps)}")

                for rep in reps:
                    spec = rep
                    if np.iscomplexobj(spec):
                        if self.mode == ExtractorMode.Magnitude:
                            spec = np.abs(spec)
                        elif self.mode == ExtractorMode.Realimag:
                             spec = np.stack([spec.real, spec.imag], axis=-1)
                        else:
                            raise ValueError('Unknown mode')
                        
                    X.append(spec.flatten())
                    y.append(gesture_name)

        X = np.array(X)
        y = np.array(y)

        print("\n=== FeatureExtractor Summary ===")
        print(f"Final X shape: {X.shape}")
        print(f"Final y shape: {y.shape}")
        print(f"Labels: {np.unique(y)}\n")

        return X, y
