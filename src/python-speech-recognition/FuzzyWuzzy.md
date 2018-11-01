# 正则表达式

re模块

强烈建议使用Python的`r`前缀，就不用考虑转义的问题了

## 使用

```
>>> import re
```



## re.compile()

编译正则表达式模式，返回一个对象的模式。（可以把那些常用的正则表达式编译成正则表达式对象，这样可以提高一点效率。）

`re.compile(pattern,flags=0)`

## re.match

re.match 尝试从字符串的起始位置匹配一个模式，如果不是起始位置匹配成功的话，match()就返回none。

**函数语法**：

```
re.match(pattern, string, flags=0)
```

## re.search

re.search 扫描整个字符串并返回第一个成功的匹配。

函数语法：

```
re.search(pattern, string, flags=0)
```

## re.match与re.search的区别

re.match只匹配字符串的开始，如果字符串开始不符合正则表达式，则匹配失败，函数返回None；而re.search匹配整个字符串，直到找到一个匹配。

*注：match和search一旦匹配成功，就是一个match object对象，而match object对象有以下方法：

- group() 返回被 RE 匹配的字符串
- start() 返回匹配开始的位置
- end() 返回匹配结束的位置
- span() 返回一个元组包含匹配 (开始,结束) 的位置
- group() 返回re整体匹配的字符串，可以一次输入多个组号，对应组号匹配的字符串。

a. group（）返回re整体匹配的字符串，
b. group (n,m) 返回组号为n，m所匹配的字符串，如果组号不存在，则返回indexError异常
c.groups（）groups() 方法返回一个包含正则表达式中所有小组字符串的元组，从 1 到所含的小组号，通常groups()不需要参数，返回一个元组，元组中的元就是正则表达式中定义的组。 

## 正则RE之flags

多个标志可以通过按位 OR-ing 它们来指定。如 re.I | re.M 。flags都有两种形式，缩写和全写都可以。

| 标志               | 含义                                                     |
| ------------------ | -------------------------------------------------------- |
| re.S(DOTALL)       | 使.匹配包括换行在内的所有字符                            |
| re.I（IGNORECASE） | 使匹配对大小写不敏感                                     |
| re.L（LOCALE）     | 做本地化识别（locale-aware)匹配，法语等                  |
| re.M(MULTILINE)    | 多行匹配，影响^和$                                       |
| re.X(VERBOSE)      | 该标志通过给予更灵活的格式以便将正则表达式写得更易于理解 |
| re.U               | 根据Unicode字符集解析字符，这个标志影响\w,\W,\b,\B       |



# fuzzywuzzy

## Requirements

- Python 2.7 or higher
- difflib
- [python-Levenshtein](https://github.com/ztane/python-Levenshtein/) (optional, provides a 4-10x speedup in String Matching, though may result in [differing results for certain cases](https://github.com/seatgeek/fuzzywuzzy/issues/128))

## Installation and Usage 

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