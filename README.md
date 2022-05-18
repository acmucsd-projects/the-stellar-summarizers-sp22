# the-stellar-summarizers-sp22

## Setup

To download and unzip SumMe dataset

```shell
./detch_dataset.py
```

To convert videos into .jpg frames in identically titled subfolders

```shell
./vid2frame.sh
```

To read and convert labels into `annotation.txt` to facilitate data loading into pytorch

```shell
./readmat.py
```

TODO:

- [ ] fix label association of the data image
- [ ] clean up `annotation.txt (possibly make it relative to each video for easier coding)
- [ ] collect all the scripts into a util folder and a unified setup.py script
- [ ] keep `requirements.txt` up to date
- [ ] make sure the scripts work cross-platform (currently macos is ok)
