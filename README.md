# Scripts for training a Kaldi speech recognition engine for Catalan. 

Scripts download and prepare datasets using the two largest speech corpora for Catalan: [Common Voice v8.0](https://commonvoice.mozilla.org/en/datasets) and [ParlamentParla](https://zenodo.org/record/5541827). Training scripts are based on [official `commonvoice` recipe](https://github.com/kaldi-asr/kaldi/tree/master/egs/commonvoice/s5). A phonetic dictionary derived from VOSK's work is provided in `dict/ca` directory. Text corpus to train the language model is derived from the training and development text plus an additional clean text corpus derived from OpenSubtitles (`corpus/CA_OpenSubtitles_clean.txt`). Evaluation is performed on test sets of both corpora. 

# Manual installation (Linux)

You need to first install Kaldi and SRILM. We provide the instructions, however you should make sure to follow the official guidelines in [Kaldi repository](https://github.com/kaldi-asr/kaldi):

```
git clone https://github.com/kaldi-asr/kaldi.git
cd kaldi/tools
extras/check_dependencies.sh
make -j 4
cd ../src
./configure --shared
make -j clean depend
make -j 8
cd ..
```

To install SRILM:

```
/opt/kaldi/tools/install_srilm.sh <name> <company> <email>
```

Then you should clone this repository under `egs` directory of `kaldi`:

```
cd egs
git clone https://github.com/CollectivaT-dev/kaldi-cat.git
cd kaldi-cat
```

Finally, make sure you also have Python 3 and installed the required modules:

```
pip install tqdm pandas
```

# Docker installation (not tested yet)

We provide a docker setup that takes care of all instalations. 

```
git clone https://github.com/CollectivaT-dev/kaldi-cat.git
cd kaldi-cat/docker
docker build -t kaldidock kaldi
```

Once the image had been built, all you have to do is interactively attach to its bash terminal via the following command:

```
$ docker run -it -v <path-to-repo>:/opt/kaldi/egs/kaldi-cat \
                 -v <path-to-cv-corpus>:/mnt \
                 --gpus all --name <container-name> <built-docker-name> bash
```

Once you finish this step, you should be in a docker container's bash terminal now to start the training.


# Training

All training scripts are inside `s5` directory: 

```
cd s5
```

If you're using GPU (and you should), make sure to flag them:

```
export CUDA_VISIBLE_DEVICES=0,1
```

To start training, all you need to do is call `run.sh` specifying a directory where to download the corpora: 

```
bash run.sh --corpora_base_path <corpus-dir>  #if running from docker <corpus-dir>=/mnt
``` 

To train toy models to see if all the process works smoothly, you can use the `subset` option. This will prepare a training dataset using only a specified number of samples:

```
bash run.sh --corpora_base_path <corpus-dir> --subset 1000
```

# Results

Evaluations are done on separately on testing portions of the two corpora. `run.sh` will print out WER scores at the end. 

```
To be published soon...
```

### Development tasks

- [x] Kaldi installation and mini-librispeech test run
- [x] Common voice dataset download scripts
- [x] Dockerfile
- [x] Text corpus from Common Voice
- [x] Builds language model
- [x] Fake phonetic dictionary
- [x] Audio data to Kaldi format
- [x] Proper phonetic dictionary
- [x] ParlamentParla download scripts
- [x] PP to Kaldi format
- [x] Combined train/dev/test1/test2
- [x] run.sh test training/evaluation
- [x] Extend LM corpus
- [x] G2P model
- [ ] Extend phonetic dictionary?
- [x] Documentation
- [ ] Docker test

