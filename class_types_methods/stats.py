import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

def imbalance_ratio(y):
    unique, counts = np.unique(y, return_counts=True)
    
    if unique.shape[0] > 2:
        raise ValueError('Provided dataset is not binary - it has more than two class different labels')
        
    ratios = np.array([counts.min(), counts.max()]) / counts.min()
    
    return f'{int(ratios[0])} : {round(ratios[1], 1)}'

def label_minority(X, y, k=5, kernel=True, unit='absolute'):
    if unit not in ['absolute', 'percent']:
        raise ValueError('Units can only take 2 values: absolute or percent')
    
    if k is not 5 and kernel is False:
        raise ValueError('Non-kernel method available only for neighbourhood of size 5')
        
    model = NearestNeighbors(n_neighbors=k+1).fit(X)
        
    unique, counts = np.unique(y, return_counts=True)
        
    if unique.shape[0] != 2:
        raise ValueError('This function works only for binary datasets. Try label_minority_mul()')
        
    minority_class = unique[counts.argmin()]
    minority_X = X[np.argwhere(y == minority_class).ravel()]
    
    neighbours_distances, neighbours_indices = model.kneighbors(minority_X)
    neighbours_distances = neighbours_distances[:, 1:]
    neighbours_indices = neighbours_indices[:, 1:]
    
    if not kernel:
        minority_X_prob = np.count_nonzero(np.take(y, neighbours_indices) == minority_class, axis=1) / k
    else:
        neighbourhood_radius = neighbours_distances[:, -1].mean()
        neighbours_distances, neighbours_indices = model.radius_neighbors(minority_X, neighbourhood_radius)
        neighbours_similarities = neighbourhood_radius - neighbours_distances
        
        minority_X_prob = np.zeros(minority_X.shape[0])
        
        for idx, (similarities, indices) in enumerate(zip(neighbours_similarities, neighbours_indices)):
            similarities_sum = 0
            
            for sim, i in zip(similarities, indices):
                if idx == i:
                    continue
                    
                if y[i] == minority_class:
                    minority_X_prob[idx] += sim
                
                similarities_sum += sim
            
            if similarities_sum != 0:
                minority_X_prob[idx] /= similarities_sum
                
    stats = {
        'safe': 0,
        'borderline': 0,
        'rare': 0,
        'outlier': 0
    }
    
    for prob in minority_X_prob:
        if prob > 0.7:
            stats['safe'] += 1
        elif prob > 0.3:
            stats['borderline'] += 1
        elif prob > 0.1:
            stats['rare'] += 1
        else:
            stats['outlier'] += 1
                
    if unit == 'percent':
        for key in stats:
            stats[key] /= (minority_X_prob.shape[0] / 100)
            stats[key] = round(stats[key], 2)
    
    return pd.DataFrame(stats, index=[minority_class])

def label_minority_mul(X, y, similarities_matrix=None, k=5, unit='absolute'):
    model = NearestNeighbors(n_neighbors=k+1).fit(X)
    labels, counts = np.unique(y, return_counts=True)
    
    if similarities_matrix is None:
        similarities_matrix = [[min(i, j) / max(i, j) for j in counts] for i in counts]
    
    all_stats = []
    
    for i, label in enumerate(labels):
        labeled_X = X[np.argwhere(y == label).ravel()]
        neighbours_distances, neighbours_indices = model.kneighbors(labeled_X)
        neighbours_distances = neighbours_distances[:, 1:]
        neighbours_indices = neighbours_indices[:, 1:]
        similarities = similarities_matrix[i]
        neighbours_classes = np.take(y, neighbours_indices)
        
        for name_idx, name in enumerate(labels):
            neighbours_classes = np.where(neighbours_classes == name, name, name_idx)
                
        neighbours_classes = neighbours_classes.astype(int)
        
        safe_levels = np.take(similarities, neighbours_classes).sum(axis=1) / k
        
        stats = {
        'safe': 0,
        'borderline': 0,
        'rare': 0,
        'outlier': 0
        }

        for sl in safe_levels:
            if sl > 0.7:
                stats['safe'] += 1
            elif sl > 0.3:
                stats['borderline'] += 1
            elif sl > 0.1:
                stats['rare'] += 1
            else:
                stats['outlier'] += 1

        if unit == 'percent':
            for key in stats:
                stats[key] /= (labeled_X.shape[0] / 100)
                
        all_stats.append(pd.DataFrame(stats, index=[label]))
            
    return pd.concat(all_stats)
