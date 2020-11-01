# AI Learning with kaikai

## Start Developing
You have to install `python3` and some other required module

### Format
Install `yapf (Yet Another Python Formatter)`

```sh
$ pip3 install yapf
```
Our format settings is [here](./.style.yapf)  
if you use `Visual Studio Code`, please set as below.  

```json
.vs/settings.json

{
  "python.formatting.provider": "yapf",
  "python.formatting.yapfArgs": [
    "--style",
    "{indent_width: 4,ARITHMETIC_PRECEDENCE_INDICATION:True,COLUMN_LIMIT:120,BLANK_LINES_AROUND_TOP_LEVEL_DEFINITION=1,based_on_style: google}"
  ]
}
```

## Learning ML  
1. 深層学習(NN)
1. オートエンコーダ　or　CNN
1. サポートベクタマシン
1. RL(Q-learning)  
