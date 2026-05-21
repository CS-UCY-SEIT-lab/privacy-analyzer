import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sentence_transformers import SentenceTransformer

_HAS_SVD = False
try:
    from sklearn.decomposition import TruncatedSVD
    _HAS_SVD = True
except Exception:
    _HAS_SVD = False


class SBERTTransformer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        batch_size=64,
        normalize=True,
        weight=1.0,
        enable_svd=False,
        svd_dim=128,
        svd_random_state=42
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.normalize = bool(normalize)
        self.weight = float(weight)
        self.enable_svd = bool(enable_svd)
        self.svd_dim = int(svd_dim)
        self.svd_random_state = int(svd_random_state)
        self._model = None
        self._svd = None

    def fit(self, X, y=None):
        self._model = SentenceTransformer(self.model_name)

        if self.enable_svd:
            if not _HAS_SVD:
                raise RuntimeError("TruncatedSVD not available.")

            emb = self._model.encode(
                list(X),
                batch_size=self.batch_size,
                show_progress_bar=False,
                normalize_embeddings=self.normalize
            )
            emb = np.asarray(emb, dtype=np.float32)

            if self.svd_dim >= emb.shape[1]:
                raise ValueError(
                    f"SBERT_SVD_DIM must be < embedding_dim ({emb.shape[1]}). Got {self.svd_dim}."
                )

            self._svd = TruncatedSVD(
                n_components=self.svd_dim,
                random_state=self.svd_random_state
            )
            self._svd.fit(emb)

        return self

    def transform(self, X):
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)

        emb = self._model.encode(
            list(X),
            batch_size=self.batch_size,
            show_progress_bar=False,
            normalize_embeddings=self.normalize
        )
        emb = np.asarray(emb, dtype=np.float32)

        if self.enable_svd and self._svd is not None:
            emb = self._svd.transform(emb).astype(np.float32)

        if self.weight != 1.0:
            emb *= self.weight

        return emb


class PrototypeSBERTSimTransformer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        batch_size=64,
        normalize=True,
        weight=1.0,
        proto_texts=None
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.normalize = bool(normalize)
        self.weight = float(weight)
        self.proto_texts = proto_texts
        self._model = None
        self._proto_emb = None

    def fit(self, X, y=None):
        self._model = SentenceTransformer(self.model_name)
        P = list(self.proto_texts) if self.proto_texts is not None else []

        proto_emb = self._model.encode(
            P,
            batch_size=self.batch_size,
            show_progress_bar=False,
            normalize_embeddings=self.normalize
        )
        self._proto_emb = np.asarray(proto_emb, dtype=np.float32)
        return self

    def transform(self, X):
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)

        emb = self._model.encode(
            list(X),
            batch_size=self.batch_size,
            show_progress_bar=False,
            normalize_embeddings=self.normalize
        )
        emb = np.asarray(emb, dtype=np.float32)

        sims = emb @ self._proto_emb.T
        sims = sims.astype(np.float32)

        if self.weight != 1.0:
            sims *= self.weight

        return sims