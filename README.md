# 省赛技术文档

队员：谈成，计松，杨雨秋



## 人脸识别

### 主要要求和任务

与操作员进行交互，识别并记忆人脸，将记忆的人脸从人群中识别并标记出来，同时检测性别，姿势肤色等等。

主要解决方案如下：

### 识别人脸

尽管opencv3中的haar级联分类器就可以检测出人脸，但是它只能检测出而不能区分不同人的人脸，有时误识别的情况比较麻烦，为了提高效果，人脸检测和识别主要用到了开源库face_recognition，github地址： https://github.com/ageitgey/face_recognition

该项目基于C++开源库dlib中的深度学习模型。注意，使用pip安装dlib可能会出问题（我们使用时也确实出了问题），建议参考文档本地编译安装。

程序逻辑：

记忆：检测人脸，当人脸数量稳定后保存脸部图片到本地，作为operator.jpg。记忆2张图片后发出提示音并转弯。

识别：逐帧分析，当检测到的人脸数量稳定后，保存当前帧到本地。分析图片，



### 人脸数据集
常用的且公开的数据集并不少，但是针对中国人或者亚洲人的并不多。我从网络上找到了CAS-PEAL-R1数据集。这是中科院CAS-PEAL数据集的一个子集，正规科研使用可以向其提交申请以获得许可和拷贝。该数据集包含了光照，远近，性别，年龄（young，middle，old），表情，头部姿势的信息，更详细的描述可以从其官网（http://www.jdl.ac.cn/peal/Home.htm）和相关论文中获得。

在备赛中，我们利用一个开源的基于keras搭建的卷积模型训练了性别识别器，由于效果不如face_recognition方案没有继续进行。考虑到face_recognition方案本质上也是通过大量图片训练而来，以后可以对它的网络进行深入了解。



### 移动

机器人（Turtlebot2）的移动利用ROS的支持完成。



### 语音

为了避免可能出现的联网问题，同时由于比赛所需的语音识别要求并不高，我们主要考虑了本地识别方案。

本地识别：

speech_recognition语音识别+fuzzywuzzy模糊识别

我们的主要思路就是利用语音识别程序识别出人说出的单词，然后将这些单词与内设的命令作比较，执行相似度最高的对应命令，如果相似度均较低则重新进入录音。

语音与命令的匹配我们初期方案为正则表达式和fuzzywuzzy模糊识别二选一，经过测试最终选择了fuzzywuzzy。fuzzywuzzy的安装和具体使用见https://github.com/seatgeek/fuzzywuzzy。



## 物品识别 

物品识别利用了tensosrlfow的object detection API（https://github.com/tensorflow/models/tree/master/research/object_detection），除了官方文档，网络上有很多相关教程和介绍，此处不再赘述，只给出我们在实验过程中的关键步骤和解决的问题。

### 关键步骤

1. 按照附录配置好环境，下载tensorflow/model和slim，添加其环境变量

```
protoc object_detection/protos/*.proto --python_out=. #**protocbuf 文件编译
python object_detection/builders/model_builder_test.py #
```

2. 准备数据集，将其整理标记为VOC2003格式。标记工具**LabelImg** (地址https://github.com/tzutalin/labelImg）。此处pyqt4/pyqt5可能会有些麻烦，根据报错一步步改即可。
3. 修改标签文件（如pascal_label_map.pbxt), 利用create_pascal_tf_record.py将数据转化为tf_record文件。
4. 选择模型，修改模型的config文件，利用model_main.py开始训练，注意根据数据集和硬件情况来进行各项参数的合理设置。
5. 利用tensorboard查看训练情况。
6. 利用export_inference_graph.py 将训练生成的文件转化成pb文件
7. 结合官方文档和例程，与opencv结合实现图片识别或者实时识别。



### 常见问题

遇到的几个重要的容易复现的问题：

1. 路径问题：环境变量没有添加成功会使得程序无法找到object detection，可以通过python的sys库来在程序内部添加。

2. 改小batch size可以改善显存不足问题导致的无法训练或意外中断。

3. 训练可能会意外中断而无法做到在无人时稳定进行，建议写个bash脚本使训练中断自动重新开始。tensorflow可以从上次训练的checkpoint处继续训练。




训练情况：

罗技摄像头拍摄的约200张图片，约2/3训练集，1/3测试集，物品种类2（口香糖，咖啡），在没有采用预训练模型的情况下训练约4小时（20k steps）后对中近距离的口香糖可以有不错的识别效果，证明该方案是可行的。





## 比赛现场

在比赛现场，我们打开了两侧辅助光源以提高面部的亮度，改善了队最边上的志愿者的识别效果。比赛中有一位女性支援者被误识别为男性，除了机器识别本身必然存在误差外也可能是识别方案是基于非亚裔人脸训练得到所导致的。

在和其他参赛队的观摩和交流中，我们了解到，有的参赛队使用darknet训练物品识别，其需要的算力小于tensorflow方案。



## 附录



### 环境：

系统：ubuntu16.04+ROS Kinetic

python包管理：anaconda3（已经添加清华镜像）

python版本：python2.7

tensorflow版本：

tensorflow-gpu            1.9.0                  

CUDA: 

cuda-9.0，cuDNN v7.1.4  for CUDA 9.0

主要的python库(仅供参考：

absl-py                   0.5.0                
dlib                      19.16.0   
face-recognition          1.2.3    
face-recognition-models   0.3.0          
Keras                     2.2.4              
labelImg                  1.7.0             
lxml                      4.2.5              
matplotlib                2.2.3             
numpy                     1.14.5            
pandas                    0.23.4            
Pillow                    5.3.0                 
protobuf                  3.6.1                 
pycocotools               2.0.0         
scikit-learn              0.20.0             
scipy                     1.1.0              
setuptools                39.1.0          
sip                       4.18                 
tensorboard               1.9.0        

fuzzywuzzy      0.17.0      

