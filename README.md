Code and sample data accompanying the paper **Modeling Multi-turn Conversation with Deep Utterance Aggregation**

## Dataset
We release E-commerce Dialogue Corpus, comprising a training data set, a development set and a test set for retrieval based chatbot. The statistics of E-commerical Conversation Corpus are shown in the following table. 

|      |Train|Val| Test         | 
| ------------- |:-------------:|:-------------:|:-------------:|
| Session-response pairs  | 1m|10k| 10k |
| Avg. positive response per session|1|1|1|  
| Min turn per session|3|3|3| 
| Max ture per session|10|10|10| 
| Average turn per session|5.51|5.48|5.64
| Average Word per utterance|7.02|6.99|7.11

The full corpus can be downloaded from https://drive.google.com/file/d/154J-neBo20ABtSmJDvm7DK0eTuieAuvw/view?usp=sharing.

## Data template
label \t conversation utterances (splited by \t) \t response

## Source Code
We also release our source code to help others reproduce our result

### Instruction 
Our code is compatible with <code>python2</code> so for all commands listed below python is <code>python2</code>

We strongly suggest you to use <code>conda</code> to control the virtual environment

* Install requirement

    <code>pip install -r requirements.txt</code>

* Pretrain word embedding

    <code>python train_word2vec.py ./ECD_sample/train embedding</code>

* Preprocess the data

    <code>python PreProcess.py --train_dataset ./ECD_sample/train --valid_dataset ./ECD_sample/valid --test_dataset ./ECD_sample/test --pretrained_embedding embedding --save_dataset ./ECD_sample/all</code>

* Train the model

    <code>bash train.sh</code>
    
    
If you use this code please cite our paper:

@inproceedings{zhang2018dua,
    title = {Modeling Multi-turn Conversation with Deep Utterance Aggregation},
    author = {Zhang, Zhuosheng and Li, Jiangtong and Zhu, Pengfei and Zhao, Hai},
    booktitle = {Proceedings of the 27th International Conference on Computational Linguistics (COLING 2018)},
    year = {2018},
}
