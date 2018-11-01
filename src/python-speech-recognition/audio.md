# 自动控制

查找文件的路径名记得要改。

语音控制在自动控制时无效，只有识别出practice指令后才开始工作。

使用前要清空outputs.

## 蜂鸣器

```
import os
duration = 1  # second
freq = 440  # Hz
os.system('play --no-show-progress --null --channels 1 synth %s sine %f' % (duration, freq))
```

必须安装`sox`

```
sudo apt install sox
```

# 语音控制

作为自动控制失败时的辅助控制。

语音识别使用的是speech_recognition

## speech_recognition

https://www.cnblogs.com/xiao1/p/6169910.html

## Fuzzywuzzy

### Requirements

- Python 2.7 or higher
- difflib
- [python-Levenshtein](https://github.com/ztane/python-Levenshtein/) (optional, provides a 4-10x speedup in String Matching, though may result in [differing results for certain cases](https://github.com/seatgeek/fuzzywuzzy/issues/128))

### Installation and Usage 

```
pip install fuzzywuzzy
pip install python-Levenshtein
```

```
>>> from fuzzywuzzy import fuzz
>>> from fuzzywuzzy import process
```

### Simple Ratio

```
>>> fuzz.ratio("this is a test", "this is a test!")
    97
```

### Partial Ratio

```
>>> fuzz.partial_ratio("this is a test", "this is a test!")
    100
```

### Token Sort Ratio

```
>>> fuzz.ratio("fuzzy wuzzy was a bear", "wuzzy fuzzy was a bear")
    91
>>> fuzz.token_sort_ratio("fuzzy wuzzy was a bear", "wuzzy fuzzy was a bear")
    100
```

### Token Set Ratio

```
>>> fuzz.token_sort_ratio("fuzzy was a bear", "fuzzy fuzzy was a bear")
    84
>>> fuzz.token_set_ratio("fuzzy was a bear", "fuzzy fuzzy was a bear")
    100
```

### Process

```
>>> choices = ["Atlanta Falcons", "New York Jets", "New York Giants", "Dallas Cowboys"]
>>> process.extract("new york jets", choices, limit=2)
    [('New York Jets', 100), ('New York Giants', 78)]
>>> process.extractOne("cowboys", choices)
    ("Dallas Cowboys", 90)
```

You can also pass additional parameters to `extractOne` method to make it use a specific scorer. A typical use case is to match file paths:

```
>>> process.extractOne("System of a down - Hypnotize - Heroin", songs)
    ('/music/library/good/System of a Down/2005 - Hypnotize/01 - Attack.mp3', 86)
>>> process.extractOne("System of a down - Hypnotize - Heroin", songs, scorer=fuzz.token_sort_ratio)
    ("/music/library/good/System of a Down/2005 - Hypnotize/10 - She's Like Heroin.mp3", 61)
```



