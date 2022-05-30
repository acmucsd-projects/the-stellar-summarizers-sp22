# the-stellar-summarizers-sp22

## Setup

Make sure you have the correct file permission and environment

```shell
conda env create -f environment.yml # creates a conda env with the name "acmai"
conda activate acmai # activate conda environment
chmod a+x ./tools/vid2frame.sh # give executing permission to the bash script
```

To download and unzip SumMe dataset

```shell
python ./tools/fetch_dataset.py
```

To convert videos into .jpg frames in identically titled subfolders

```shell
./tools/vid2frame.sh
```

To read and convert labels into `annotation.txt` to facilitate data loading into pytorch

```shell
python ./tools/readmat.py
```

If you want to use colab or other platforms that require you to upload files on to a remote server the following allows you to convert the .jpg frames from each video into single `frames.hdf5` file.

```shell
python ./tools/hdf5.py
```

Note: our dataloader is NOT yet compatible with this file format

TODO:

- [ ] implement loss history plotting
- [ ] clean up `annotation.txt (possibly make it relative to each video for easier coding)
- [ ] keep `requirements.txt` / `environment.yml` up to date
- [x] collect all the scripts into a util folder / unified script
- [x] look into a different file format of faster dataset upload to gdrive (hdf5?)
- [x] write scripts to standardize video resolution (downsample) (explore options: opencv? pytorch? ffmpeg?) 
- [x] make sure the scripts work cross-platform (currently macos is ok)
- [x] fix label association of the data image
