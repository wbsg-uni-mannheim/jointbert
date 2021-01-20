# JointBERT

This repository contains the code and data download links to reproduce the experiments of the PVLDB 2021 paper "Dual-Objective Fine-Tuning of BERT for Entity Matching" by Ralph Peeters and Christian Bizer. [More information](https://www.uni-mannheim.de/dws/news/paper-accepted-for-pvldb-2021-dual-objective-fine-tuning-of-bert-for-entity-matching/) about JointBERT and its use cases.

* **Requirements**

    [Anaconda3](https://www.anaconda.com/products/individual)

    Please keep in mind that the code is not optimized for portable or even non-workstation devices. Some of the scripts require large amounts of RAM (64GB+) and GPUs. It is advised to use a powerful workstation or server when experimenting with some of the large files.

    The code has only been used and tested on Linux (Manjaro, Ubuntu, CentOS) computers.

* **Building the conda environment**

    To build the exact conda environment used for the experiments, navigate to the project root folder where the file *jointbert.yml* is located and run ```conda env create -f jointbert.yml```
    
    Furthermore you need to install the project as a package. To do this, activate the jointbert environment with ```conda activate jointbert```, navigate to the root folder of the project, and run ```pip install -e .```

* **Downloading the raw data files/models**

    Navigate to the *src/data/* folder and run ```python download_datasets.py``` to automatically download the files into the correct locations.
    You can find the data at *data/raw/* and the trained models used for generating explanations at *src/productbert/saved/models*

    If you are only interested in the separate datasets, you can download the [WDC LSPC datasets](http://webdatacommons.org/largescaleproductcorpus/v2/index.html#toc6) and the [deepmatcher splits](https://github.com/anhaidgroup/deepmatcher/blob/master/Datasets.md) for the abt-buy, dblp-scholar and company datasets on the respective websites. The data and ground truth for the monitors dataset can be found [here](http://di2kg.inf.uniroma3.it/datasets.html#downloads). The training, validation and test sets we used in the paper were derived from this data and can be downloaded separately [here](http://data.dws.informatik.uni-mannheim.de/largescaleproductcorpus/data/v2/repo-download/monitor-datasets.zip).

* **Processing the data**

    To prepare the data for the experiments, run the following scripts in that order. Make sure to navigate to the respective folders first.
    
    1. *src/processing/preprocess/preprocess_corpus.py*
    2. *src/processing/preprocess/preprocess_ts_gs.py*
    3. *src/processing/preprocess/preprocess_deepmatcher_datasets.py*
    4. *src/processing/preprocess/preprocess_di2kg.py*
    5. *src/processing/process-bert/process_to_bert.py*
    6. *src/processing/process-magellan/process_to_magellan.py*
    7. *src/processing/process-wordcooc/process-to-wordcooc.py*
    
    optional: prepare data for Ditto input: *src/processing/process-ditto/process_to_ditto.py*
    
* **Running the baseline experiments**

    Run the following scripts to replicate the baseline experiments:
    * **Magellan**:
        Navigate to *src/models/magellan/* and run the script *run_magellan.py*
    * **Word Coocurrence**:
    Navigate to *src/models/wordcooc/* and run the script *run_wordcooc.py*
    * **Deepmatcher**:
    Navigate to *src/models/deepmatcher* and run any of the scripts *train_\*.py*.
    
    Result files can subsequently be found in the *reports* folder.

* **Running the BERT experiments**

    Navigate to *src/productbert/*
    This project is based on a <a target="_blank" href="https://github.com/victoresque/pytorch-template/">PyTorch template project</a> It is suggested to read the respective github readme to understand how to train models and possible input commands.
    * **Fine-Tuning**:
    The folder *src/productbert* contains bash scripts to run all of the experiments including the learning rate sweeps. Run any of the bash scripts titled *train_\*.sh* and append the id of the gpu you want to use, e.g. ```bash train_computers_bert.sh 0```

    * **Evaluating a trained model on a test set**:
    This is done by providing a config containing configuration parameters for the test run. Additionally you need to provide the checkpoint that should be used for testing. An example would be ```python test.py --device 0 -c configs/BERT/config_computers_small_test.json --resume saved/models/pathto/model/model_best.pth```
    
    The results of the BERT experiments can then be found in *src/productbert/saved/log* and the respective model checkpoints in *src/productbert/saved/models*.
    
    **NOTE**: When adjusting the batch size in any of the configs, make sure to also adjust the number of accumulation steps, as the combination of both constitutes the actual batch size.

* **Explanations**

    Jupyter Notebooks used to generate explanations and corresponding swarmplots can be found at *src/productbert*. The swarmplot images can be found at *data/processed/explain_labeling/pictures*.

* **Ditto**

    The code for the Ditto framework can be found [here](https://github.com/megagonlabs/ditto).
    
* **MWPD test set results for all WDC training set sizes**

	| Matching Challenge                  | Training Size | Word Co-oc | Magellan | Deepmatcher |  BERT | RoBERTa | Ditto | JointBERT |
	|-------------------------------------|:-------------:|:----------:|:--------:|:-----------:|:-----:|:-------:|:--------:|:---------:|
	| Unseen products - high similarity   | xlarge        |    48.00   |   27.89  |    52.71    | 84.53 |  83.18  |   69.97  |   59.92   |
	|                                     | large         |    42.86   |   35.54  |    55.53    | 75.19 |  85.33  |   75.57  |   60.45   |
	|                                     | medium        |    47.89   |   31.91  |    53.51    | 73.14 |  74.48  |   68.22  |   63.42   |
	|                                     | small         |    36.67   |   30.49  |    45.91    | 63.91 |  71.40  |   61.66  |   55.76   |
	| Unseen products - low similarity    | xlarge        |    11.43   |   59.04  |    59.49    | 76.92 |  77.72  |   65.78  |   58.91   |
	|                                     | large         |    21.05   |   61.21  |    57.12    | 73.01 |  78.18  |   70.68  |   58.85   |
	|                                     | medium        |    34.62   |   49.23  |    58.27    | 73.00 |  72.67  |   69.23  |   63.04   |
	|                                     | small         |    32.14   |   51.64  |    60.88    | 68.00 |  71.70  |   63.12  |   60.29   |
	| Seen products - introduced typos    | xlarge        |    54.01   |   19.79  |    54.89    | 71.21 |  85.49  |   83.49  |   89.08   |
	|                                     | large         |    54.01   |   31.92  |    59.43    | 71.49 |  85.90  |   80.88  |   91.29   |
	|                                     | medium        |    66.67   |   58.12  |    56.46    | 72.60 |  83.55  |   76.02  |   77.80   |
	|                                     | small         |    63.01   |   49.62  |    43.12    | 55.20 |  73.08  |   64.81  |   66.05   |
	| Seen products - dropped tokens      | xlarge        |    78.79   |   50.35  |    73.32    | 87.62 |  89.88  |   90.28  |   93.23   |
	|                                     | large         |    75.00   |   58.11  |    68.08    | 90.09 |  89.49  |   89.04  |   92.07   |
	|                                     | medium        |    79.52   |   80.00  |    66.93    | 93.61 |  92.83  |   92.23  |   94.74   |
	|                                     | small         |    75.78   |   74.37  |    65.06    | 88.04 |  89.99  |   92.08  |   93.98   |
	| Seen products - very hard cases     | xlarge        |    61.22   |   11.00  |    77.07    | 89.04 |  88.51  |   94.12  |   95.55   |
	|                                     | large         |    56.00   |   24.49  |    64.48    | 82.61 |  89.69  |   87.65  |   94.82   |
	|                                     | medium        |    48.48   |   18.54  |    39.66    | 77.78 |  80.49  |   72.83  |   75.63   |
	|                                     | small         |    40.00   |   16.65  |    27.35    | 46.24 |  66.56  |   52.96  |   46.73   |
	| Mix of hard non-matches and matches | xlarge        |    77.93   |   58.48  |    78.83    | 84.08 |  86.32  |   85.30  |   84.24   |
	|                                     | large         |    75.38   |   59.44  |    76.52    | 81.86 |  86.70  |   83.96  |   84.52   |
	|                                     | medium        |    66.89   |   58.31  |    66.71    | 80.02 |  83.81  |   79.94  |   78.97   |
	|                                     | small         |    58.69   |   55.24  |    60.18    | 74.62 |  80.64  |   75.52  |   71.69   |
	| Full MWPD test set                   | xlarge        |    69.65   |   48.23  |    71.53    | 82.58 |  86.20  |   83.96  |   83.35   |
	|                                     | large         |    67.22   |   52.60  |    71.45    | 80.84 |  86.61  |   83.15  |   83.67   |
	|                                     | medium        |    64.91   |   56.22  |    62.98    | 80.27 |  83.92  |   79.61  |   79.09   |
	|                                     | small         |    57.88   |   52.31  |    55.42    | 71.51 |  79.34  |   73.74  |   70.98   |
	
--------

Project based on the [cookiecutter data science project template](https://drivendata.github.io/cookiecutter-data-science/). #cookiecutterdatascience

PyTorch Project based on the [PyTorch template project](https://github.com/victoresque/pytorch-template/) by [Victor Huang](https://github.com/victoresque).
