from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

class Preprocessor():
    def __init__(self, n_components=100):
        self.n_components = n_components
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=n_components)
        self._fitted = False

    def fit(self, X):
        print("=== Preprocessor.fit ===")
        print(f"Input X shape: {X.shape}")

        X_scaled = self.scaler.fit_transform(X)
        print("Scaling done.")

        self.pca.fit(X_scaled)
        print(f"PCA fitted with {self.n_components} components.")

        self._fitted = True
        return self

    def transform(self, X):
        if not self._fitted:
            raise RuntimeError("Preprocessor must be fitted before calling transform().")

        print("=== Preprocessor.transform ===")
        print(f"Input X shape: {X.shape}")

        X_scaled = self.scaler.transform(X)
        print("Scaling applied.")

        X_pca = self.pca.transform(X_scaled)
        print(f"PCA transform done. Output shape: {X_pca.shape}")

        return X_pca
    
    def fit_transform(self, X):
        print("=== Preprocessor.fit_transform ===")
        print(f"Input X shape: {X.shape}")

        X_scaled = self.scaler.fit_transform(X)
        print("Scaling done.")

        X_pca = self.pca.fit_transform(X_scaled)
        print(f"PCA done. Shape after PCA: {X_pca.shape}")

        self._fitted = True
        return X_pca
