# Tensorflow 2.0 英汉翻译

## 数据集

https://cms.unov.org/UNCorpus/en/DownloadOverview

http://www.manythings.org/anki/

https://www.statmt.org/wmt17/translation-task.html

https://www.statmt.org/wmt18/translation-task.html

https://www.statmt.org/wmt19/translation-task.html

下载并整理后的数据集在 https://yun.yusanshi.com/TF_datasets/en_zh.zip （约33w条）。

## 运行

1. 修改`config.py`中的`NUM_EXAMPLES`、`BATCH_SIZE`。前者表示要训练的数据量，填入`None`表示训练全部数据；
2. `pip install jieba sklearn`；
3. `python main.py`。若程序发现`MODEL_PATH`中有数据，说明已经有训练好的模型，程序会直接恢复之，从而直接利用该模型来翻译。加入`force_retrain=True`参数来强制重新训练。

## 示例

> 本人的电脑比较辣鸡（GTX 1050Ti），显存 4 G，数据量稍大就爆显存（OOM），下面的测试是把`NUM_EXAMPLES`调到 20000 的结果，如果把 33w 的测试数据全部用上，翻译效果应该会好得多。

```
Load model from saved_model successfully.
{'At the time, There were no native English speakers teaching in any public school.': "KeyError('speakers')",
 'Do you love me?': '你爱我吗？',
 'Experience is the best teacher.': '经验是最好的老师。',
 'He is getting better day by day.': '他一天一天地好转。',
 'How are you?': '你好吗？',
 'I am convinced that things will change for the better.': '我相信要做一次。',
 'I ate a hamburger.': "KeyError('hamburger')",
 'I cannot agree with you on the matter.': '在这件事上，我无法赞同你。',
 "I don't like you.": '我不喜欢你。',
 'I know what you mean.': '我知道你的意思。',
 'I know you.': '我知道你。',
 'I like football.': '我喜欢足球。',
 'I love you.': '我爱您。',
 "It's me!": '太这是！',
 'Please make three copies of each page.': '请把每一页复印三份。',
 'Some people clung to tree branches for several hours to avoid being washed away by the floodwaters.': "KeyError('clung')",
 "That's my book.": '那是我的书。',
 'The elephant likes painting.': '大象有长生活。',
 'The wind blew too hard for them to play in the park.': '今天早上很长的噪音惊吓了。',
 'What is the baby doing?': '这是谁？',
 'What is your name?': '您叫什么名字？',
 "What's your name?": '您叫什么名字？',
 'You are wonderful.': '你很无聊。'}
```

