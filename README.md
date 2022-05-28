# the-stellar-summarizers-sp22

## Setup

Make sure you have the correct file permission and environment

```shell
conda env create -f environment.yml # creates a conda env with the name "acmai"
conda activate acmai # activate conda environment
chmod a+x ./vid2frame.sh # give executing permission to the bash script
```

To download and unzip SumMe dataset

```shell
python ./fetch_dataset.py
```

To convert videos into .jpg frames in identically titled subfolders

```shell
./vid2frame.sh
```

To read and convert labels into `annotation.txt` to facilitate data loading into pytorch

```shell
python ./readmat.py
```

TODO:

- [ ] write scripts to standardize video resolution (downsample) (explore options: opencv? pytorch? ffmpeg?) 
- [ ] implement loss history plotting
- [ ] clean up `annotation.txt (possibly make it relative to each video for easier coding)
- [ ] collect all the scripts into a util folder and a unified setup.py script
- [ ] keep `requirements.txt` / `environment.yml` up to date
- [ ] look into a different file format of faster data loading (hd5?)
- [x] make sure the scripts work cross-platform (currently macos is ok)
- [x] fix label association of the data image
