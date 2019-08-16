# Tensorflow 2.0 英汉翻译

## 数据集

https://cms.unov.org/UNCorpus/en/DownloadOverview

http://www.manythings.org/anki/

https://www.statmt.org/wmt17/translation-task.html

https://www.statmt.org/wmt18/translation-task.html

https://www.statmt.org/wmt19/translation-task.html

下载并整理后的数据集在 https://yun.yusanshi.com/TF_datasets/en_zh.zip（约33w条）。

## 运行

1. 修改`config.py`中的`NUM_EXAMPLES`、`BATCH_SIZE`。前者表示要训练的数据量，填入`None`表示训练全部数据；
2. `pip install jieba sklearn`；
3. `python main.py`。若程序发现`MODEL_PATH`中有数据，说明已经有训练好的模型，程序会直接恢复之，从而直接利用该模型来翻译。加入`force_retrain=True`参数来强制重新训练。
