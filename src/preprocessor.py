from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np

class Preprocessor():
    def __init__(self, n_components=100):
        self.n_components = n_components
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=n_components)
        self._fitted = False

    def fit(self, X):
        print('=== Preprocessor.fit ===')
        print(f'Input X shape: {X.shape}')

        X_scaled = self.scaler.fit_transform(X)
        print('Scaling done.')

        self.pca.fit(X_scaled)
        print(f'PCA fitted with {self.n_components} components.')

        self._fitted = True
        return self

    def transform(self, X):
        if not self._fitted:
            raise RuntimeError('Preprocessor must be fitted before calling transform().')

        print('=== Preprocessor.transform ===')
        print(f'Input X shape: {X.shape}')

        X_scaled = self.scaler.transform(X)
        print('Scaling applied.')

        X_pca = self.pca.transform(X_scaled)
        print(f'PCA transform done. Output shape: {X_pca.shape}')

        return X_pca
    
    def fit_transform(self, X):
        print('=== Preprocessor.fit_transform ===')
        print(f'Input X shape: {X.shape}')

        X_scaled = self.scaler.fit_transform(X)
        print('Scaling done.')

        X_pca = self.pca.fit_transform(X_scaled)
        print(f'PCA done. Shape after PCA: {X_pca.shape}')

        self._fitted = True
        return X_pca

    def plot_elbow(self, X, max_components=300, save_as_file = False, include_title = True):
        print('=== PCA Elbow Method ===')

        X_scaled = self.scaler.fit_transform(X)

        pca_full = PCA(n_components=max_components)
        pca_full.fit(X_scaled)

        explained = pca_full.explained_variance_ratio_
        cumulative = np.cumsum(explained)

        n_used = self.n_components
        var_used = cumulative[n_used - 1]  

        plt.figure(figsize=(8, 5))
        plt.plot(cumulative, linewidth=2, label='Skumulowany udział wariancji')

        plt.axhline(0.90, linestyle='--', color='gray', label='90% wariancji')
        plt.axhline(0.95, linestyle='--', color='red', label='95% wariancji')

        plt.axvline(n_used, linestyle=':', color='blue', linewidth=2,
                    label=f'Wykorzystane składowe = {n_used}')
        plt.scatter(n_used, var_used, color='blue', zorder=5)

        plt.text(
            n_used + 5,
            var_used,
            f'{var_used:.2%}',
            verticalalignment='center',
            fontsize=10,
            color='blue'
        )

        plt.xlabel('Liczba składowych głównych')
        plt.ylabel('Skumulowany udział wariancji')
        if include_title:
            plt.title('Dobór liczby składowych głównych metodą łokcia')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        if save_as_file:
            plt.savefig('pca_elbow.png', dpi=300, bbox_inches='tight')

        plt.show()
