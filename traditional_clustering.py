"""
EN
================

ENK-Means、EN、EN，ENDBSCANEN。

EN：
1. EN（MDS）
2. EN（kEN）
3. EN
4. EN
"""

import numpy as np
from typing import Tuple, List, Dict, Optional, Callable
from sklearn.cluster import (
    KMeans,
    AgglomerativeClustering,
    SpectralClustering,
)
from sklearn.manifold import MDS
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')



def select_k_for_kmeans(
    D: np.ndarray,
    k_range: Tuple[int, int] = (2, 10),
    method: str = 'silhouette',
    n_components: int = 10,
    random_state: int = 42,
    verbose: bool = False,
) -> Tuple[int, List[Tuple[int, float]]]:
    """
    ENK-MeansENkEN
    
    EN
    ----
    D : np.ndarray, shape=(N, N)
        EN
    k_range : Tuple[int, int]
        kEN (min_k, max_k)
    method : str
        EN：'silhouette'（EN）EN 'elbow'（EN）
    n_components : int
        MDSEN
    random_state : int
        EN
    verbose : bool
        EN
    
    EN
    ----
    best_k : int
        ENkEN
    scores : List[Tuple[int, float]]
        ENkEN
    """
    if D.ndim != 2 or D.shape[0] != D.shape[1]:
        raise ValueError(f"D must be a square matrix, got shape {D.shape}")
    
    N = D.shape[0]
    min_k, max_k = k_range
    
    min_k = max(2, min_k)
    max_k = min(N - 1, max_k)
    
    if min_k >= max_k:
        return min_k, [(min_k, 0.0)]
    
    if verbose:
        print(f"[K-Means] ENMDSEN {n_components} EN...")
    
    n_components = min(n_components, N - 1)
    mds = MDS(
        n_components=n_components,
        dissimilarity='precomputed',
        random_state=random_state,
        n_init=10,
    )
    try:
        X = mds.fit_transform(D)
    except Exception as e:
        if verbose:
            print(f"⚠️ MDSEN: {e}, EN")
        n_components = min(5, N - 1)
        mds = MDS(
            n_components=n_components,
            dissimilarity='precomputed',
            random_state=random_state,
        )
        X = mds.fit_transform(D)
    
    if verbose:
        print(f"✅ MDSEN，EN: {X.shape}")
    
    scores = []
    best_k = min_k
    best_score = -np.inf if method == 'silhouette' else np.inf
    
    for k in range(min_k, max_k + 1):
        try:
            kmeans = KMeans(
                n_clusters=k,
                random_state=random_state,
                n_init=10,
                max_iter=300,
            )
            labels = kmeans.fit_predict(X)
            
            unique_labels = np.unique(labels)
            if len(unique_labels) < 2:
                if verbose:
                    print(f"  k={k}: EN，EN")
                continue
            
            if method == 'silhouette':
                score = silhouette_score(X, labels)
                scores.append((k, score))
                if score > best_score:
                    best_score = score
                    best_k = k
                if verbose:
                    print(f"  k={k}: silhouette={score:.4f}")
            
            elif method == 'elbow':
                inertia = kmeans.inertia_
                scores.append((k, inertia))
                if len(scores) >= 3:
                    if k == min_k + 2:
                        best_k = k
                        best_score = inertia
                    else:
                        prev_inertia = scores[-2][1]
                        drop_rate = (prev_inertia - inertia) / prev_inertia
                        if drop_rate > 0.1:
                            if inertia < best_score:
                                best_k = k
                                best_score = inertia
                if verbose:
                    print(f"  k={k}: inertia={inertia:.4f}")
        
        except Exception as e:
            if verbose:
                print(f"  k={k}: EN - {e}")
            continue
    
    if not scores:
        best_k = min_k
        if verbose:
            print(f"⚠️ ENkEN，EN k={best_k}")
    
    if verbose:
        print(f"✅ EN k={best_k} (method={method})")
    
    return best_k, scores


def kmeans_clustering(
    D: np.ndarray,
    k: Optional[int] = None,
    k_range: Tuple[int, int] = (2, 10),
    n_components: int = 10,
    random_state: int = 42,
    verbose: bool = False,
) -> Tuple[np.ndarray, int, Dict]:
    """
    ENK-MeansEN
    
    EN
    ----
    D : np.ndarray, shape=(N, N)
        EN
    k : int, optional
        EN。ENNone，ENk
    k_range : Tuple[int, int]
        ENkEN
    n_components : int
        MDSEN
    random_state : int
        EN
    verbose : bool
        EN
    
    EN
    ----
    labels : np.ndarray
        EN (0, 1, 2, ..., k-1)
    k : int
        EN
    info : Dict
        EN：
        - 'method': 'kmeans'
        - 'k': EN
        - 'n_components': MDSEN
        - 'silhouette_score': EN
        - 'inertia': K-MeansEN
        - 'selection_scores': kEN
    """
    if D.ndim != 2 or D.shape[0] != D.shape[1]:
        raise ValueError(f"D must be a square matrix, got shape {D.shape}")
    
    N = D.shape[0]
    
    n_components = min(n_components, N - 1)
    mds = MDS(
        n_components=n_components,
        dissimilarity='precomputed',
        random_state=random_state,
        n_init=10,
    )
    
    if verbose:
        print(f"[K-Means] MDSEN: {N}×{N} -> {N}×{n_components}")
    
    try:
        X = mds.fit_transform(D)
    except Exception as e:
        if verbose:
            print(f"⚠️ MDSEN: {e}")
        raise RuntimeError(f"MDSEN: {e}")
    
    selection_scores = []
    if k is None:
        if verbose:
            print(f"[K-Means] ENk (EN: {k_range})...")
        k, selection_scores = select_k_for_kmeans(
            D=D,
            k_range=k_range,
            method='silhouette',
            n_components=n_components,
            random_state=random_state,
            verbose=verbose,
        )
    else:
        k = int(k)
        if k < 2 or k >= N:
            raise ValueError(f"k must be in [2, {N-1}], got {k}")
    
    if verbose:
        print(f"[K-Means] EN，k={k}...")
    
    kmeans = KMeans(
        n_clusters=k,
        random_state=random_state,
        n_init=10,
        max_iter=300,
    )
    labels = kmeans.fit_predict(X)
    
    unique_labels = np.unique(labels)
    actual_k = len(unique_labels)
    
    silhouette = 0.0
    if actual_k >= 2:
        try:
            silhouette = float(silhouette_score(X, labels))
        except:
            silhouette = 0.0
    
    inertia = float(kmeans.inertia_)
    
    info = {
        'method': 'kmeans',
        'k': actual_k,
        'n_components': n_components,
        'silhouette_score': silhouette,
        'inertia': inertia,
        'selection_scores': selection_scores,
        'centroids': kmeans.cluster_centers_,
    }
    
    if verbose:
        print(f"✅ [K-Means] EN: k={actual_k}, silhouette={silhouette:.4f}")
    
    return labels, actual_k, info



def select_n_clusters_agglomerative(
    D: np.ndarray,
    max_clusters: int = 10,
    linkage: str = 'average',
    random_state: int = 42,
    verbose: bool = False,
) -> Tuple[int, List[Tuple[int, float]]]:
    """
    EN
    
    EN
    ----
    D : np.ndarray, shape=(N, N)
        EN
    max_clusters : int
        EN
    linkage : str
        EN：'average', 'complete', 'single', 'ward'
    random_state : int
        EN（EN，EN）
    verbose : bool
        EN
    
    EN
    ----
    best_n : int
        EN
    scores : List[Tuple[int, float]]
        EN
    """
    if D.ndim != 2 or D.shape[0] != D.shape[1]:
        raise ValueError(f"D must be a square matrix, got shape {D.shape}")
    
    N = D.shape[0]
    max_clusters = min(max_clusters, N - 1)
    min_clusters = 2
    
    if min_clusters >= max_clusters:
        return min_clusters, [(min_clusters, 0.0)]
    
    scores = []
    best_n = min_clusters
    best_score = -np.inf
    
    for n in range(min_clusters, max_clusters + 1):
        try:
            agg = AgglomerativeClustering(
                n_clusters=n,
                metric='precomputed',
                linkage=linkage,
            )
            labels = agg.fit_predict(D)
            
            unique_labels = np.unique(labels)
            if len(unique_labels) < 2:
                continue
            
            score = silhouette_score(D, labels, metric='precomputed')
            scores.append((n, score))
            
            if score > best_score:
                best_score = score
                best_n = n
            
            if verbose:
                print(f"  n={n}: silhouette={score:.4f}")
        
        except Exception as e:
            if verbose:
                print(f"  n={n}: EN - {e}")
            continue
    
    if not scores:
        best_n = min_clusters
        if verbose:
            print(f"⚠️ EN，EN n={best_n}")
    
    if verbose:
        print(f"✅ EN n={best_n} (linkage={linkage})")
    
    return best_n, scores


def agglomerative_clustering(
    D: np.ndarray,
    n_clusters: Optional[int] = None,
    max_clusters: int = 10,
    linkage: str = 'average',
    verbose: bool = False,
) -> Tuple[np.ndarray, int, Dict]:
    """
    EN
    
    EN
    ----
    D : np.ndarray, shape=(N, N)
        EN
    n_clusters : int, optional
        EN。ENNone，EN
    max_clusters : int
        EN
    linkage : str
        EN：'average', 'complete', 'single', 'ward'
    verbose : bool
        EN
    
    EN
    ----
    labels : np.ndarray
        EN (0, 1, 2, ..., k-1)
    k : int
        EN
    info : Dict
        EN
    """
    if D.ndim != 2 or D.shape[0] != D.shape[1]:
        raise ValueError(f"D must be a square matrix, got shape {D.shape}")
    
    N = D.shape[0]
    
    selection_scores = []
    if n_clusters is None:
        if verbose:
            print(f"[EN] EN (EN: {max_clusters})...")
        n_clusters, selection_scores = select_n_clusters_agglomerative(
            D=D,
            max_clusters=max_clusters,
            linkage=linkage,
            verbose=verbose,
        )
    else:
        n_clusters = int(n_clusters)
        if n_clusters < 2 or n_clusters >= N:
            raise ValueError(f"n_clusters must be in [2, {N-1}], got {n_clusters}")
    
    if verbose:
        print(f"[EN] EN，n_clusters={n_clusters}, linkage={linkage}...")
    
    agg = AgglomerativeClustering(
        n_clusters=n_clusters,
        metric='precomputed',
        linkage=linkage,
    )
    labels = agg.fit_predict(D)
    
    unique_labels = np.unique(labels)
    actual_k = len(unique_labels)
    
    silhouette = 0.0
    if actual_k >= 2:
        try:
            silhouette = float(silhouette_score(D, labels, metric='precomputed'))
        except:
            silhouette = 0.0
    
    info = {
        'method': 'agglomerative',
        'k': actual_k,
        'linkage': linkage,
        'silhouette_score': silhouette,
        'selection_scores': selection_scores,
    }
    
    if verbose:
        print(f"✅ [EN] EN: k={actual_k}, silhouette={silhouette:.4f}")
    
    return labels, actual_k, info



def spectral_clustering(
    D: np.ndarray,
    n_clusters: Optional[int] = None,
    max_clusters: int = 10,
    sigma: Optional[float] = None,
    random_state: int = 42,
    verbose: bool = False,
) -> Tuple[np.ndarray, int, Dict]:
    """
    EN
    
    EN
    ----
    D : np.ndarray, shape=(N, N)
        EN
    n_clusters : int, optional
        EN。ENNone，EN
    max_clusters : int
        EN
    sigma : float, optional
        EN。ENNone，EN
    random_state : int
        EN
    verbose : bool
        EN
    
    EN
    ----
    labels : np.ndarray
        EN (0, 1, 2, ..., k-1)
    k : int
        EN
    info : Dict
        EN
    """
    if D.ndim != 2 or D.shape[0] != D.shape[1]:
        raise ValueError(f"D must be a square matrix, got shape {D.shape}")
    
    N = D.shape[0]
    
    if sigma is None:
        valid_distances = D[D > 0]
        if valid_distances.size > 0:
            sigma = float(np.median(valid_distances))
        else:
            sigma = 1.0
    
    if verbose:
        print(f"[EN] EN，sigma={sigma:.4f}...")
    
    S = np.exp(-D**2 / (2 * sigma**2))
    np.fill_diagonal(S, 0.0)
    
    selection_scores = []
    if n_clusters is None:
        if verbose:
            print(f"[EN] EN (EN: {max_clusters})...")
        
        max_clusters = min(max_clusters, N - 1)
        min_clusters = 2
        
        best_n = min_clusters
        best_score = -np.inf
        
        for n in range(min_clusters, max_clusters + 1):
            try:
                spectral = SpectralClustering(
                    n_clusters=n,
                    affinity='precomputed',
                    random_state=random_state,
                )
                labels = spectral.fit_predict(S)
                
                unique_labels = np.unique(labels)
                if len(unique_labels) < 2:
                    continue
                
                score = silhouette_score(S, labels, metric='precomputed')
                selection_scores.append((n, score))
                
                if score > best_score:
                    best_score = score
                    best_n = n
                
                if verbose:
                    print(f"  n={n}: silhouette={score:.4f}")
            
            except Exception as e:
                if verbose:
                    print(f"  n={n}: EN - {e}")
                continue
        
        n_clusters = best_n
    else:
        n_clusters = int(n_clusters)
        if n_clusters < 2 or n_clusters >= N:
            raise ValueError(f"n_clusters must be in [2, {N-1}], got {n_clusters}")
    
    if verbose:
        print(f"[EN] EN，n_clusters={n_clusters}...")
    
    spectral = SpectralClustering(
        n_clusters=n_clusters,
        affinity='precomputed',
        random_state=random_state,
    )
    labels = spectral.fit_predict(S)
    
    unique_labels = np.unique(labels)
    actual_k = len(unique_labels)
    
    silhouette = 0.0
    if actual_k >= 2:
        try:
            silhouette = float(silhouette_score(S, labels, metric='precomputed'))
        except:
            silhouette = 0.0
    
    info = {
        'method': 'spectral',
        'k': actual_k,
        'sigma': sigma,
        'silhouette_score': silhouette,
        'selection_scores': selection_scores,
    }
    
    if verbose:
        print(f"✅ [EN] EN: k={actual_k}, silhouette={silhouette:.4f}")
    
    return labels, actual_k, info



def traditional_clustering(
    D: np.ndarray,
    method: str = 'kmeans',
    n_clusters: Optional[int] = None,
    **kwargs,
) -> Tuple[np.ndarray, int, Dict]:
    """
    EN
    
    EN
    ----
    D : np.ndarray, shape=(N, N)
        EN
    method : str
        EN：'kmeans', 'agglomerative', 'spectral'
    n_clusters : int, optional
        EN。ENNone，EN
    **kwargs
        EN，EN
    
    EN
    ----
    labels : np.ndarray
        EN
    k : int
        EN
    info : Dict
        EN
    """
    if method == 'kmeans':
        return kmeans_clustering(D, k=n_clusters, **kwargs)
    elif method == 'agglomerative':
        return agglomerative_clustering(D, n_clusters=n_clusters, **kwargs)
    elif method == 'spectral':
        return spectral_clustering(D, n_clusters=n_clusters, **kwargs)
    else:
        raise ValueError(f"Unknown method: {method}. Choose from: 'kmeans', 'agglomerative', 'spectral'")



if __name__ == "__main__":
    np.random.seed(42)
    N = 50
    D = np.random.rand(N, N) * 5.0
    D = (D + D.T) / 2
    np.fill_diagonal(D, 0)
    
    print("=" * 60)
    print("EN")
    print("=" * 60)
    
    print("\n1. ENK-MeansEN...")
    labels_kmeans, k_kmeans, info_kmeans = kmeans_clustering(
        D, k=None, k_range=(2, 10), verbose=True
    )
    print(f"   EN: k={k_kmeans}, silhouette={info_kmeans['silhouette_score']:.4f}")
    
    print("\n2. EN...")
    labels_agg, k_agg, info_agg = agglomerative_clustering(
        D, n_clusters=None, max_clusters=10, verbose=True
    )
    print(f"   EN: k={k_agg}, silhouette={info_agg['silhouette_score']:.4f}")
    
    print("\n3. EN...")
    labels_spec, k_spec, info_spec = spectral_clustering(
        D, n_clusters=None, max_clusters=10, verbose=True
    )
    print(f"   EN: k={k_spec}, silhouette={info_spec['silhouette_score']:.4f}")
    
    print("\n4. EN...")
    labels, k, info = traditional_clustering(
        D, method='kmeans', n_clusters=5, verbose=False
    )
    print(f"   EN: k={k}, method={info['method']}")
    
    print("\n✅ EN！")
