#!/usr/bin/env bash

# Copyright   2014  Johns Hopkins University (author: Daniel Povey)
#             2017  Luminar Technologies, Inc. (author: Daniel Galvez)
#             2017  Ewald Enzinger
#	            2022  Col·lectivaT, SCCL
# Apache 2.0

# Downloads ParlamentParla corpus from Zenodo and untars it
# Adapted from egs/mini_librispeech/s5/local/download_and_untar.sh (commit 1cd6d2ac3a935009fdc4184cb8a72ddad98fe7d9)

remove_archive=false

if [ "$1" == --remove-archive ]; then
  remove_archive=true
  shift
fi

if [ $# -ne 2 ]; then
  echo "Usage: $0 [--remove-archive] <data-base>"
  echo "e.g.: $0 /export/data/"
  echo "With --remove-archive it will remove the archive after successfully un-tarring it."
fi

data=$1

dev_url=https://zenodo.org/record/5541827/files/clean_dev.tar.gz
test_url=https://zenodo.org/record/5541827/files/clean_test.tar.gz
train_url=https://zenodo.org/record/5541827/files/clean_train.tar.gz

echo datadir: $data

if [ ! -d "$data" ]; then
  echo "$0: no such directory $data"
  exit 1;
fi

if [ -f $data/.complete ]; then
  echo "$0: data was already successfully extracted, nothing to do."
  exit 0;
fi

for url in $dev_url $test_url $train_url; do
  echo downloading: $url
  filename=$(basename $url)
  filepath="$data/$filename"

  if [ -f $filepath ]; then
    size=$(/bin/ls -l $filepath | awk '{print $5}')
    echo TAR size: $size
  fi

  if [ ! -f $filepath ]; then
    if ! which wget >/dev/null; then
      echo "$0: wget is not installed."
      exit 1;
    fi
    echo "$0: downloading data from $url.  This may take some time, please be patient."

    echo wget --no-check-certificate $url
    cd $data
    if ! wget --no-check-certificate $url; then
      echo "$0: error executing wget $url"
      exit 1;
    fi
  fi

  cd $data

  echo tar -xzf $filepath
  if ! tar -xzf $filepath; then
    echo "$0: error un-tarring archive $filepath"
    exit 1;
  fi

  echo "$0: Successfully downloaded and un-tarred $filepath"

  if $remove_archive; then
    echo "$0: removing $filepath file since --remove-archive option was supplied."
    rm $filepath
  fi

done

touch $data/$lang/.complete



