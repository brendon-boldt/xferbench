#!/usr/bin/env bash


echo This script is intended to outline the steps to reproduce the main results in the paper.  In all likelihood, you will have to run this script line-by-line to troubleshoot any errors.  The script assumes a GNU/Linux environment. | fold -s
echo
echo -n Press Enter to continue.
read foo

set -xe



### Environment setup ###
conda create --name xferbench --file environment.txt

set +x
source activate xferbench
set -x



### Setting up data ###
wget http://patient.lti.cs.cmu.edu:12001/xferbench-paper-data.tar.gz
tar xf xferbench-paper-data.tar.gz
# zstd is required to decompress some of the baseline datasets.
type zstd
# Decompress all zst archives
find data/ -name "*.zst" -exec zstd -dk {} \;

for script in hierarchical_parens.py random_data.py prepare_yao.py wikipedia.py; do
  python xferbench/scripts/$script
done

### XferBench ###
for target in $(ls data/eval); do
  python -m xferbench clm_init_model -t $target
done
for lang in $(ls data/baselines); do
  python -m xferbench clm_train_base_model -s $lang
  for target in $(ls data/eval); do
    python -m xferbench clm_tune_model -s $lang -t $target
  done
done

python -m xferbench clm_analysis --save save-clm/
# Results in save-clm/analysis


### MT experiments ###
python -m xferbench mt_init_model -t en-fr
for lang in $(ls data/baselines); do
  python -m xferbench mt_train_base_model -s $lang -c MtWmt
  # Full, Frozen, Reduced
  for cfg in MtWmt MtFreeze MtLL; do
    python -m xferbench mt_tune_model -s $lang -t en-fr -c $cfg
    # Results will be in save-mt/$lang/en-fr-$cfg
  done
done
python -m xferbench mt_analysis --save save-mt/
# Results in save-mt/analysis
cp save-clm/analysis/clm.pkl save-mt/analysis
python -m xferbench correlation_analysis --save save-mt/
