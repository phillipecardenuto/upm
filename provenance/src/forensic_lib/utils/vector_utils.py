
import numpy as np
import torch.nn.functional as F

def normalize_vector(
    vec: np.ndarray,
    axis: int = -1,
    order: int = 2
) -> np.ndarray:
    """
    Normalize a vector.
    If you pass a Matrix, each line of the matrix will be 
    considerated as a different vector and normalized as it.
    """
    l2 = np.atleast_1d(np.linalg.norm(vec, order, axis))
    l2[l2==0] = 1
    return vec / np.expand_dims(l2, axis)





def similarity(embeddings_1, embeddings_2):
    """
    Get Similarity between two vectors
    """
    normalized_embeddings_1 = F.normalize(embeddings_1, p=2)
    normalized_embeddings_2 = F.normalize(embeddings_2, p=2)
    return torch.matmul(
        normalized_embeddings_1, normalized_embeddings_2.transpose(0, 1)
    )

