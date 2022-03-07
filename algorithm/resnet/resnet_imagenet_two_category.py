# 导入依赖
import time

import torch
from mlxtend.evaluate import accuracy
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter, writer
from torchvision import transforms, datasets, models
import matplotlib.pyplot as plt
import numpy as np

import logging
from logging.handlers import RotatingFileHandler

# 分别为train, val, test定义transform
image_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(size=300, scale=(0.8, 1.1)),  # 功能：随机长宽比裁剪原始图片, 表示随机crop出来的图片会在的0.08倍至1.1倍之间
        transforms.RandomRotation(degrees=10),  # 功能：根据degrees随机旋转一定角度, 则表示在（-10，+10）度之间随机旋转
        transforms.ColorJitter(0.4, 0.4, 0.4),  # 功能：修改亮度、对比度和饱和度
        transforms.RandomHorizontalFlip(),  # 功能：水平翻转
        transforms.CenterCrop(size=256),  # 功能：根据给定的size从中心裁剪，size - 若为sequence,则为(h,w)，若为int，则(size,size)
        transforms.ToTensor(),  # numpy --> tensor
        # 功能：对数据按通道进行标准化（RGB），即先减均值，再除以标准差
        transforms.Normalize([0.485, 0.456, 0.406],  # mean
                             [0.229, 0.224, 0.225])  # std
    ]),

    'val': transforms.Compose([
        transforms.Resize(300),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],  # mean
                             [0.229, 0.224, 0.225])  # std
    ]),

    'test': transforms.Compose([
        transforms.Resize(300),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],  # mean
                             [0.229, 0.224, 0.225])  # std
    ])
}


# 加载数据集

# 数据集所在目录路径
data_dir = ''
# train路径
train_dir = data_dir + 'train/'
# val路径
val_dir = data_dir + 'val/'
# test路径
test_dir = data_dir + 'test/'

def set_data_dir(real_data_dir):
    data_dir = real_data_dir
    return data_dir

# 从文件中读取数据
datasets = {
    'train': datasets.ImageFolder(train_dir, transform=image_transforms['train']),  # 读取train中的数据集，并transform
    'val': datasets.ImageFolder(val_dir, transform=image_transforms['val']),  # 读取val中的数据集，并transform
    'test': datasets.ImageFolder(test_dir, transform=image_transforms['test'])  # 读取test中的数据集，并transform
}

# 定义BATCH_SIZE
BATCH_SIZE = 128  # 每批读取128张图片
# BATCH_SIZE = 512  # 每批读取128张图片

# DataLoader : 创建iterator, 按批读取数据
dataloaders = {
    'train': DataLoader(datasets['train'], batch_size=BATCH_SIZE, shuffle=True),  # 训练集
    'val': DataLoader(datasets['val'], batch_size=BATCH_SIZE, shuffle=True),  # 验证集
    'test': DataLoader(datasets['test'], batch_size=BATCH_SIZE, shuffle=True)  # 测试集
}

# 创建label的键值对
LABEL = dict((v, k) for k, v in datasets['train'].class_to_idx.items())


# 定义日志路径
log_path = 'logdir/'

# 定义函数：获取tensorboard writer
def tb_writer():
    timestr = time.strftime("%Y%m%d_%H%M%S")  # 时间格式
    writer = SummaryWriter(log_path + timestr)  # 写入日志
    return writer

# 定义图片显示方法
def imshow(img):
    img = img / 2 + 0.5  # 逆正则化
    np_img = img.numpy()  # tensor --> numpy
    plt.imshow(np.transpose(np_img, (1, 2, 0)))  # 改变通道顺序
    plt.show()


# 记录错误分类的图片
def misclassified_images(pred, writer, target, images, output, epoch, count=10):
    misclassified = (pred != target.data)  # 判断是否一致
    for index, image_tensor in enumerate(images[misclassified][:count]):
        img_name = 'Epoch:{}-->Predict:{}-->Actual:{}'.format(epoch, LABEL[pred[misclassified].tolist()[index]],
                                                              LABEL[target.data[misclassified].tolist()[index]])
        writer.add_image(img_name, image_tensor, epoch)

# 自定义池化层
class AdaptiveConcatPool2d(nn.Module):
    def __init__(self, size=None):
        super(AdaptiveConcatPool2d, self).__init__()
        size = size or (1, 1) # kernel大小
        # 自适应算法能够自动帮助我们计算核的大小和每次移动的步长
        self.avgPooling = nn.AdaptiveAvgPool2d(size) # 自适应平均池化
        self.maxPooling = nn.AdaptiveMaxPool2d(size) #最大池化

    def forward(self, x):
        # 拼接avg 和 max
        return torch.cat([self.maxPooling(x), self.avgPooling(x)], dim=1)

# 迁移学习
def create_model():
    # 获取预训练模型
    model = models.resnet50(pretrained=True)
    # 冻结模型参数
    for param in model.parameters():
        param.requires_grad = False

    # 替换最后两层：池化层和全连接层
    model.avgpool = nn.AdaptiveAvgPool2d() #池化层
    model.fc = nn.Sequential(   #全连接层
        nn.Flatten(), #拉平
        nn.BatchNorm1d(4096), #加速神经网络的收敛过程， 提高训练过程中的稳定性
        nn.Dropout(0.5), #丢掉部分神经网络，防止过拟合
        nn.ReLU(), #激活函数
        nn.BatchNorm1d(512),
        nn.Dropout(0.5),
        nn.Linear(512, 2), # 二分类只有两个输出
        nn.LogSoftmax(dim=1) #损失函数：将input转换成概率分布式的形式，输出2个概率
    )
    return model

# 定义训练函数
def train_val(model, device, train_loader, val_loader, criterion, optimizer, epoch):
    model.train()
    total_loss = 0
    val_loss = 0
    val_acc = 0
    for batch_id, (images, labels) in enumerate(train_loader):
        # 部署到device上
        images, labels = images.to(device), labels.to(device)
        # 梯度置0
        optimizer.zero_grad()
        # 模型输出
        outputs = model(images)
        # 计算损失
        loss = criterion(outputs, labels)
        # 反向传播
        loss.backward()
        # 更新参数
        optimizer.step()
        # 累计损失
        total_loss += loss.item() * images.size(0)

    # 平均训练损失
    train_loss = total_loss / len(train_loader.dataset)
    # 写入到writer中
    writer.add_scalar("Train Loss", train_loss, epoch )
    # 写入到磁盘
    writer.flush()

    model.eval()
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images) # 前向传播输出
            loss = criterion(outputs, labels) #损失
            val_loss += loss.item() * images.size(0) #累计损失
            _,pred = torch.max(outputs, 1) #获取最大概率的索引
            corrent = pred.eq(labels.view_as(pred)) # 返回： tensor([True, False, ... , Fasle])
            accuracy = torch.mean(corrent.type(torch.FloatTensor)) # 准确率
            val_acc += accuracy.item() * images.size(0) # 累计准确率

        # 平均验证损失
        val_loss = val_loss / len(val_loader.dataset)
        # 平均验证准确率
        val_acc = val_acc / len(val_loader.dataset)

    return train_loss, val_loss, val_acc

# 定义测试函数
def test(model, device, test_loader, criterion, epoch, writer):
    model.eval()
    total_loss = 0
    correct = 0.0 # 正确数
    with torch.no_grad():
        for batch_id, (images, labels) in enumerate(test_loader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _,pred = torch.max(outputs, 1)
            correct += pred.eq(labels.view_as(pred)).sum().item() # 累计正确预测的数
            misclassified_images(pred, writer, labels, images, outputs, epoch)
    # 平均损失
    avg_loss = total_loss / len(test_loader.dataset)
    # 计算准确率
    accuracy = 100 * correct / len(test_loader.dataset)
    # 将test的结果写入writer中
    writer.add_scalar("Test Loss", total_loss, epoch )
    writer.add_scalar("ACcuracy", accuracy, epoch )
    writer.flush()

    return avg_loss, accuracy

# 定义训练过程
def train_epochs(model, device, dataloaders, criterion, optimizer, epochs, writer):
    logging.info("{0:>15} | {1:>15} | {2:>15} | {3:>15} | {4:>15} | {5:>15}".format('Epoch', 'Train Loss', 'val_loss', 'val_acc',
                                                                           'Test Loss', 'Test_acc') )
    # 初始最小的损失
    best_loss = np.inf
    # 开始训练、测试
    for epoch in range(epochs):
        # 训练，return: loss
        train_loss, val_loss, val_acc = train_val(model, device, dataloaders['train'], dataloaders['val'], optimizer,
                                                  criterion, epoch, writer)
        # 测试，return: loss + accuracy
        test_loss, test_acc = test(model, device, dataloaders['test'], criterion, epoch, writer)
        # 判断损失是否最小
        if test_loss < best_loss:
            best_loss = test_loss  # 保存最小损失
            # 保存模型
            torch.save(model.state_dict(), 'model.pth')
        # 输出结果
        logging.info("{0:>15} | {1:>15} | {2:>15} | {3:>15} | {4:>15} | {5:>15}".format(epoch, train_loss, val_loss, val_acc,
                                                                                 test_loss, test_acc))
        writer.flush()


def train_new_model():
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    model = create_model().to(device)
    # criterion = nn.CrossEntropyLoss()
    criterion = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001) # 数据量较少，不适合使用Adam
    epochs = 10
    writer = tb_writer()
    train_epochs(model, device, dataloaders, criterion, optimizer, epochs, writer)





