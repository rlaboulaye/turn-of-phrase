# Turn of Phrase: Conversational Structure Pre-training

This repository includes a python implementation of Turn of Phrase pre-training, a pre-training method used to further pre-train a MLM-pre-trained transformer-based language model for tasks that benefit from conversational discourse understanding.

## Dependencies

The dependencies are listed in *requirements.txt*. Assuming [pip](https://pip.pypa.io/en/stable/) is already installed, they can be installed via
```
$ pip install -r requirements.txt
```

## Usage

The model can be pre-trained by running
```
$ python pretrain.py
```

It can be evaluated on discourse act classification by running
```
$ python evaluate_discourse.py
```

It can be evaluated on downstream tasks from [convokit](https://convokit.cornell.edu/) by running
```
$ python evaluate_downstream.py
```

Further arguments for the above programs can be found in their respective files.
