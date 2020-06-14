# Indoor WiFi Positioning

This study uses user-generated WiFi fingerprints to derive locations at different granularity levels. 

For optimal results, each room location should have a *distinct* set of *repeated* access points with *varying* signal intensity. 

## File structure

- `data/`: UJIIndoorLoc data (original and subsets)
- `notebooks/`: Jupyter notebooks for classification 
- `results/`: K-Means clustering results
- `utils/`: helper functions
- `visualisation/`: visualisation plots

## Results

- Building classification: 99.88\% of test accuracy
- Floor classification: 88.56\% to 93.44\% of test accuracy
- Room classification: varied accuracy scores

