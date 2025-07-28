import numpy as np
from scipy.sparse import csr_matrix

def load_data():
    # Load the data
    X_train_resampled = np.load('data/processed/X_train_resampled.npz')
    X_train_resampled = csr_matrix((X_train_resampled['data'], 
                                     X_train_resampled['indices'], 
                                     X_train_resampled['indptr']),
                                    shape=X_train_resampled['shape'])

    y_train = np.load('data/processed/y_train_resampled.npy')

    X_test = np.load('data/processed/X_test_processed.npz')
    X_test = csr_matrix((X_test['data'], 
                         X_test['indices'], 
                         X_test['indptr']),
                        shape=X_test['shape'])

    y_test = np.load('data/processed/y_test.npy')
    
    return X_train_resampled, y_train, X_test, y_test
