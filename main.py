import kagglehub

# Download latest version
path = kagglehub.dataset_download("kuantinglai/ntut-4k-drone-photo-dataset-for-human-detection")

print("Path to dataset files:", path)