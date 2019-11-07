Runs the model on Pang and Lee's movie review dataset (MR in the paper).

### Directory structure
CI1: Model trained with 
	static vector : Variable
	task-specific vector: Random

CI2: Model trained with 
	static vector : Constant
	task-specific vector: word2vec

CI3: Model trained with 
	static vector : Variable
	task-specific vector: word2vec

conv_net_sentence_tanh: Model trained with tanh as the activation function instead of relu. **Resulted in a better accuracy.**

conv_net_sigmoid: Model trained with sigmoid as the activation function

conv_net_sentenceh: Model trained with 50 hidden layers (instead of 100).

### Requirements
Code is written in Python (2.7) and requires Theano (0.7).

Using the pre-trained `word2vec` vectors will require downloading the binary file from
https://code.google.com/p/word2vec/


### Data Preprocessing
To process the raw data, run

```
python process_data.py <path>
```

where path points to the word2vec binary file (i.e. `GoogleNews-vectors-negative300.bin` file). 
This will create a pickle object called `mr.p` in the same folder, which contains the dataset in the right format.


### Running the models (CPU)
Example commands:

```
THEANO_FLAGS=mode=FAST_RUN,device=cpu,floatX=float32 python conv_net_sentence.py -nonstatic -rand
THEANO_FLAGS=mode=FAST_RUN,device=cpu,floatX=float32 python conv_net_sentence.py -static -word2vec
THEANO_FLAGS=mode=FAST_RUN,device=cpu,floatX=float32 python conv_net_sentence.py -nonstatic -word2vec
```

This will run the CNN-rand, CNN-static, and CNN-nonstatic models respectively in the paper.


### Example output
```
epoch: 1, training time: 219.72 secs, train perf: 81.79 %, val perf: 79.26 %
epoch: 2, training time: 219.55 secs, train perf: 82.64 %, val perf: 76.84 %
epoch: 3, training time: 219.54 secs, train perf: 92.06 %, val perf: 80.95 %
```
