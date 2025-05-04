import h5py

filepath = "BraTS2020/content/data/volume_9_slice_78.h5"
with h5py.File(filepath, 'r') as f:
    print("Keys in the file:", list(f.keys()))