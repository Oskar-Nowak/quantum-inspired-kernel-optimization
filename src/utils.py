import matplotlib.pyplot as plt
import numpy as np

def plot_spectrogram(spec, title='Spectrogram'):
    plt.figure(figsize=(10, 6))
    plt.imshow(np.abs(spec),aspect='auto', cmap='jet')
    plt.colorbar()
    plt.title(title)
    plt.xlabel('Time bins')
    plt.ylabel('Frequency bins')
    plt.show()