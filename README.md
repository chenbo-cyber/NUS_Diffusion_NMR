### Environment Setup
Conda with Python 3.9.21 is recommended.

The torch-related libraries can be installed using the following command:
`pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu118`

Then install the dependencies listed in `requirements`:
`pip install -r requirements.txt`

### 2D Data Processing (using GB1 data as an example):

Training: You can use the `train256.py` file to train a one-dimensional undersampled reconstruction model on simulated data (a pre-trained weight file `sr_NUSNMR_250930_081820` is also provided in the code repository).

Testing: Use the `take_sample2D` file directly. The example file can be used to test .mat data or raw fid data; the results will be automatically saved to the data folder, named `label_spec.mat` and `recon_spec.mat` respectively.

### 3D Data Processing: In `deal_64` (using A3DK08 as an example):

Training: A 2D undersampled reconstruction model can be trained on simulated data using the `train64.py` file (a pre-trained weight file `sr_NUSNMR_251212_073902` is also provided in the code repository).

Testing: The A3DK08 folder contains the raw .fid data.

1. In the nmrpipe environment, use `proc_dirct` to convert the .fid data to the hypercomplex format A3DK08_laebl.ft1.

2. Reconstruct A3DK08_laebl.ft1 using the `take_sample3D_automatic.py` file.

3. In the nmrpipe environment, run `proc_indirct` (the internal code of `proc_indirct` needs to be modified to fit the label and recon files; actually, only the filename needs to be changed) to perform zero-padding and windowing on the data.

4. Finally, slice the final data into contours and save it (because including all slices would introduce a lot of noise).
