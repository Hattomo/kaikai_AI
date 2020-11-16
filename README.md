# AI Learning with kaikai

## Start Developing
You have to install `python3`, and module written in [./requirements.txt](./requirements.txt).

To run cnn with [THE MNIST DATABASE of handwritten digits](http://yann.lecun.com/exdb/mnist/),
please download the data set from [here](https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz) and locate the file to `dataset/input/`.  

Create `out` folder to save graph, and numpy files.

### Format
Install `yapf (Yet Another Python Formatter)`.

```sh
$ pip3 install yapf
```
Our format settings is [here](./.style.yapf)  
if you use `Visual Studio Code`, please set as below.  

```json
.vscode/settings.json

{
  "python.formatting.provider": "yapf",
  "python.formatting.yapfArgs": [
    "--style",
    "{indent_width: 4,ARITHMETIC_PRECEDENCE_INDICATION:True,COLUMN_LIMIT:120,BLANK_LINES_AROUND_TOP_LEVEL_DEFINITION=1,based_on_style: google}"
  ]
}
```

### Tests
We use `pytest` to test our code.  
Please install and run.  
The tests run with `python3.8`.  

```sh
$ pip3 install pytest
$ pytest -v
```
We add tests when we confirm that the theory is correct.

### Comments
We write comments actively, because we can see more easily.  
Comments are written English or Japanese sentence, and required in situations like below.
- Attention needed
- Multiple tasks in one function
- Complex tasks

## Road Map & Status
|No.|Road Map|Status|
|:--:|:--:|:--:|
|1|DNN|🐥|
|2|CNN|🐣|
|3|RNN|🥚|
|4|RL(Q-learning)|🥚|
|4|Auto Encoder|🥚|
|5|SVM|🥚|

🍳: No plan yet  
🥚: Not started  
🐣: Early Developing stage 🌱  
🐥: Tests stage 🧪  
🐤: v1 Completed 🎉  
🐔: Advanced Developing stage🔨 (v2)  