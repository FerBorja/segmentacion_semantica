X = np.load("preprocessed_data/X_batch_0.npy")
Y = np.load("preprocessed_data/Y_batch_0.npy")
print("Forma de X:", X.shape)  # Debe ser (N, 128, 128, 4)
print("Forma de Y:", Y.shape)  # Debe ser (N, 128, 128, 1)