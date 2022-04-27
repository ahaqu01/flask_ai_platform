# Flask Ai Platform

## 1.描述
### 1.1 简介
- 该项目旨在基于flask-restful搭建一个简易的AI平台：包括自动化AI训练、AI服务，主要为CV方向，后续可能会添加大数据、自然语言处理等方向模型。
- 该项目提供一个基础平台，提供部分检测识别训练实现模块，主要的AI服务、无代码训练以远程接口提供给该平台，该系统为分布式系统。
### 1.2 使用者描述
- 该项目分为三种使用者：管理员、第三方开发者、客户用户。
- 管理员：负责系统管理；包含一个超级管理员。
- 第三方开发者：可在该平台通过用户界面，进行低代码开发，只需要按照要求上传所需要训练的数据，就可以得到相应的模型；
- 客户用户：可使用该平台的模型进行AI检测、识别、建模等工作。并提供一些指定硬件自动化部署的一键式解决方案。

## 2.环境依赖
### 2.1 运行环境
- Ubuntu server 18.04, conda 4.5.11, python 3.7.11, cuda-10.1
```
# https://pytorch.org/
conda install pytorch torchvision torchaudio cudatoolkit=10.1 -c pytorch
```
### 2.2 依赖环境
#### 2.2.1 python 依赖
- 详见requirement.txt
- 安装：
```dockerfile
pip install -r requirement.txt
```
#### 2.2.2 其他依赖
- mysql 5.7、redis 4.0.9

## 3.运行方法
- 1.启动mysql、redis
```
# mysql先创建数据库
mysql -u mysql_username -p # 输入密码登入
> create database dbname charset=utf8; # 创建数据库

# 启动redis-server
sudo /etc/init.d/redis-server start
```
- 2.数据迁移
    - python manage.py db init
    - python manage.py db migrate
    - python manage.py db upgrade
- 3.运行系统
    - 方法一：python manage.py runserver -h xxx.xxx.xxx.xxx -p xxxx (-d -r --threaded)
    - 方法二(高并发运行，用于生产环境)：gunicorn -c guncorn.conf manage:app  # 参见doc/Note.md
    
## 4.使用说明
### 4.1 用户注册、登入
- 注册
- 登入

### 4.2 智能检测、识别、分割、生成等服务
#### 4.2.1 densenet
- 初级版本：当前项目 algorithm/densenet
    - 功能：基于imageNet数据集的检测、识别
    - postman使用例子：
```
POST=> hostname:port/ai_service/predict/
输入参数：image(file)=>选择要上传的图片
返回结果：
{
    "object_name": "xxx"
}
```
- 远程接口：当前项目： remote_ai_interface/cv
    - 待接口设计后接入系统
#### 4.2.2 yolov5
- 本地版本：当前项目 algorithm/pytorch_yolov5
    - 功能：基于yolov5的图像、视频检测、识别，训练自己的模型等
    - 第三方依赖：
        - pip install paho-mqtt   # 适合嵌入式（物联网硬件）通信使用的mqtt
        - pip install thop        # 用于统计模型的 FLOPs 和 参数量
- 远程接口： 当前项目： remote_ai_interface/cv
    - 待接口设计后接入系统
- 数据集制作方法
    - voc数据集参考网上
    - 待完善
#### 4.2.3 resnet
- 本地版本： 当前项目 algorithm/rensenet
    - 功能：
        - 基于imageNet数据集的迁移学习训练、验证、预测
        - 基于imageNet数据集迁移学习的二分类训练、验证、预测，如肺部、心脏、胃部影像是否感染、损伤、病变，生产的同一件产品是否有损坏（划痕、缺陷）
- 远程接口：当前项目： remote_ai_interface/cv
    - 待接口设计后接入系统
    
- 数据集制作方法
    - 待完善

#### 4.2.4 mask rcnn
- 本地版本： 当前项目 algorithm/mask_rcnn
    - 功能：
        - 基于现有模型进行预测和实例分割
        - 基于coco数据集迁移学习训练、验证、预测分割，代码待加入
- 远程接口：当前项目： remote_ai_interface/cv
    - 待接口设计后接入系统
- 数据集制作方法
    - 待制作

### 4.3 自动化训练

### 4.4 模型转换部署

## 5.项目结构
