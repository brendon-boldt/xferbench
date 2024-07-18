![XferBench logo](./assets/logo.svg)
XferBench
=========

XferBench is a benchmark/evaluation metric for emergent language corpora (not
familiar? see [Lazaridou and Baroni, 2020](https://arxiv.org/abs/2006.02419))
presented and published at [NAACL
2024](https://aclanthology.org/2024.naacl-long.82/).  This metric measures the
overall _quality_ of an emergent language using deep transfer learning: the
better that an emergent language serves as pretraining data for a human
language-based downstream task, the more similar it is to human language from
a neural network's perspective.  Below we include a digram describing how
XferBench works.

![Benchmark architecture](./assets/benchmark-chart.svg)


## Quick Start

Looking to directly reproduce the results of the NAACL 2024 paper?  See
`./reproduce.sh`.  Continue reading for a description of the steps to run
XferBench in general.

Install the conda environment.

    conda create --name xferbench --file environment.txt

Download the target language data (required to run XferBench) with:

    python xferbench/scripts/wikipedia.py eval

Ensure your data is in the JSON lines format where each row is an array of
integer tokens, representing an utterance from the emergent communication
system.  For example,

    [3, 14, 15, 9]
    [2, 6, 5, 35]
    [8, 9, 7, 9, 38, 3]

Then, run the benchmark on the corpus file.

    python -m xferbench some-directory/my_dataset/corpus.jsonl

Output will be in `save-clm/xferbench-my_dataset_corpus/results.json`.


### Where to get data

Don't have any data to try XferBench with?  No problem!  A tarball of the data
used in the original data can be downloaded
[here](http://patient.lti.cs.cmu.edu:12001/xferbench-paper-data.tar.gz).  We
encourage you, though, to check out
[ELCC](https://huggingface.co/datasets/bboldt/elcc) which is a more
comprehensive collection of emergent language corpora with accompanying
metadata.


## FAQs

- Does XferBench support multiple GPUs?
    - No.  While HuggingFace should do it for free, it doesn't work.  Plus, the
      typical workflow of XferBench means it is typically easier to run it on
      different inputs in parallel instead of speeding up a single run.
- Why is CUDA unexpectedly OOMing?
    - Many things could be to blame, but one known problem is if HuggingFace
      can see multiple GPUs and tries to use more than one of them.
      `xferbench/__main__.py` has some code to prevent this, but if you are
      say, importing modules from XferBench, this will not take effect.  Using
      `CUDA_VISIBLE_DEVICES=0 python ...` is the easiest way to prevent this.

## Citation

If using this work in research please cite the paper:

    @inproceedings{boldt-mortensen-2024-xferbench,
        title = "{X}fer{B}ench: a Data-Driven Benchmark for Emergent Language",
        author = "Boldt, Brendon  and
          Mortensen, David",
        editor = "Duh, Kevin  and
          Gomez, Helena  and
          Bethard, Steven",
        booktitle = "Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers)",
        month = jun,
        year = "2024",
        address = "Mexico City, Mexico",
        publisher = "Association for Computational Linguistics",
        url = "https://aclanthology.org/2024.naacl-long.82",
        pages = "1475--1489",
    }
