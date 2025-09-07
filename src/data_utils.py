import numpy as np
import pandas as pd

def load_spotify_data(path, seed=42, test_ratio=0.2):
    df = pd.read_csv(path)
    num_songs = df.shape[0]
    num_features = df.shape[1] - 1  # Exclude the 'track_name' column

    subset_df = df[df["genre"].isin(["Pop", "Classical"])] #getting only pop and classical songs
    subset_df = subset_df.copy()
    subset_df["label"] = subset_df["genre"].apply(lambda x: 1 if x == "Pop" else 0) # label 1 for pop, 0 for classical

    x = subset_df[["liveness", "loudness"]].to_numpy()
    y = subset_df["label"].to_numpy()

    pop_idx = np.flatnonzero(y == 1) #index for every pop song
    clas_idx = np.flatnonzero(y == 0) #inex for every classical song

    random_gen = np.random.default_rng(seed) #random generator with seed 42

    random_gen.shuffle(pop_idx) #shuffling the pop indexes
    random_gen.shuffle(clas_idx) #shuffling the classical indexes

    n_pop_test = int(test_ratio * len(pop_idx)) #20% of pop songs for testing
    n_clas_test = int(test_ratio * len(clas_idx)) #20% of classical songs for testing

    pop_test, pop_train = pop_idx[:n_pop_test], pop_idx[n_pop_test:] #splitting the pop indexes into test and train
    clas_test, clas_train = clas_idx[:n_clas_test], clas_idx[n_clas_test:] #splitting the classical indexes into test and train

    train_idx = np.concatenate([pop_train, clas_train])
    test_idx  = np.concatenate([pop_test, clas_test])

    X_train = x[train_idx]
    y_train = y[train_idx]

    X_test  = x[test_idx]
    y_test  = y[test_idx]

    return X_train, y_train, X_test, y_test

