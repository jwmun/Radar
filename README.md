# Radar

FMCW, OFDM radar simulator tensorflow code


## Requirements
```
Python3, Numpy, Tensorflow
```

## Dataset

* Our dataset can be downloaded from the below link

* FMCW data ([link](https://drive.google.com/file/d/18s95iyC_ZovvPxSe75rgdS2Q8LFSE_j8/view?usp=sharing))

* MIMO data ([link](https://drive.google.com/file/d/1ep1i7wUamg4g1EkyKo_Ls9DM_BlLAIZS/view?usp=sharing))


## Running the code
* The final deep learning output needs to be denormalized to return to its original state since the signal strength has changed due to the normalization process in preprocess.py. Refer to the output.py file of each FMCW and OFDM folder to denormalize.(FMCW signal is 8 channel and 75 chirps)
* Put the data you received above into the appropriate path and modify the code.
* If the interference strength is large, a median filter is required.
### FMCW

```
Python3 run_transformer.py
```
### MIMO
```
Python3 run_transformer.py
```


## Acknowledgments


