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

## Citation
If you use any part of this code in your research, please cite our paper:
```
@inproceedings{mun2020automotive,
  title={Automotive Radar Signal Interference Mitigation Using RNN with Self Attention},
  author={Mun, Jiwoo and Ha, Seokhyeon and Lee, Jungwoo},
  booktitle={ICASSP 2020-2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={3802--3806},
  year={2020},
  organization={IEEE}
}
```
## Acknowledgments

This work is in part supported by Bio-Mimetic Robot Research Center Funded by Defense
Acquisition Program Administration, Agency for Defense Development (UD190018ID),
MSIT-IITP grant (No.2019-0-01367, BabyMind), Grant(UD190031RD) from Defense
Acquisition Program Administration(DAPA) and Agency for Defense Development(ADD),
INMAC, and BK21-plus.

## License

MIT License