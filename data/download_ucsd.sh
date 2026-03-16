#!/bin/bash
# Download UCSD Ped2 dataset

echo "Downloading UCSD Ped2 dataset..."

mkdir -p data/ucsd
cd data/ucsd

# Download
wget http://www.svcl.ucsd.edu/projects/anomaly/UCSD_Anomaly_Dataset.tar.gz

# Extract
tar -xzf UCSD_Anomaly_Dataset.tar.gz

# Cleanup
rm UCSD_Anomaly_Dataset.tar.gz

echo "UCSD Ped2 dataset downloaded to data/ucsd/"
