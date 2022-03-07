# 导入依赖包
import glob
import shutil
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt

import torch
from PIL import Image
from torch import nn, optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision import transforms, models
from torchvision.datasets import ImageFolder
from torchvision.utils import make_grid
import torch.nn.functional as F


# 训练集文件夹路径
train_path = ""

# 测试集文件夹路径
test_path = ""

# 获取训练集文件夹路径
def get_train_path(train_path_param):
    train_path = train_path_param
    return train_path

# 获取测试集文件夹路径
def get_test_path(test_path_param):
    test_path = test_path_param
    return test_path


# 默认显示前5个类别的文件夹路径
glob.glob(train_path)[:5]


# 按照比例： train 70%, val 20%, test 10% 创建文件夹，将每个类别的图片按此比例分配

# 第一步：分别创建train, val, test下各个类别的文件夹

classes_path = sorted(glob.glob(train_path))  # 所有类别的文件夹路径
categories = ['train', 'val', 'test']  # 分别创建3个文件夹
new_data_dir = Path('NEW_DATA')  # 新数据集目录

for cat in categories:
    for path in classes_path:
        class_name = path[-5:]  # 类别名称即为文件夹名称的最后5位数字
        (new_data_dir / cat / class_name).mkdir(parents=True, exist_ok=True)

# 第二步：将原始train下的各个类别的图片按照比例保存到新建的文件夹下
class_indices = np.arange(len(classes_path))  # 所有类别对应的索引

for i, class_idx in enumerate(class_indices):
    img_paths = np.array(sorted(glob.glob(f'{classes_path[int(class_idx)]}/*.ppm')))  # 图片路径
    class_name = classes_path[i][-5:]  # 类别名称
    # 打乱图片路径
    np.random.shuffle(img_paths)
    # 索引分割 70%, 20%, 10%
    paths_split = np.split(img_paths, indices_or_sections=[int(0.7 * len(img_paths)), int(0.9 * len(img_paths))])
    for ds, pathes in zip(categories, paths_split):
        for path in pathes:
            shutil.copy(path, f'{new_data_dir}/{ds}/{class_name}')

# 根据路径，加载图片
def load_img(img_path, resize=True):
    image = cv2.imread(img_path)  # 根据路径，加载图片
    image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)  # 颜色转换
    if resize == True:
        image = cv2.resize(image, (64, 64))
    return image

# 根据路径，显示图片
def show_img(img_path):
    img = load_img(img_path)  # 加载图片
    plt.imshow(img)  # 显示图片
    plt.axis('off')  # 关闭坐标轴

# 随机获取一张图片的路径，并显示
class_0_paths = np.array(sorted(glob.glob(f'{classes_path[0]}/*.ppm')))  # 索引0对应的类别的所有图片路径
random_img_path = np.random.choice(class_0_paths)
print(random_img_path)
show_img(random_img_path)

# 数据预处理
mean_nums = [0.485, 0.456, 0.406]
std_nums = [0.229, 0.224, 0.225]
transform = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(size=256),  # 随机裁剪
        transforms.RandomRotation(degrees=15),  # 随机旋转
        transforms.RandomHorizontalFlip(),  # 随机水平翻转
        transforms.ToTensor(),  # 转换为tensor
        transforms.Normalize(mean_nums, std_nums)
    ]),

    'val': transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize(mean_nums, std_nums)
    ]),

    'test': transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize(mean_nums, std_nums)
    ])
}

# 数据文件对象
image_folders = {
    k: ImageFolder(f'{new_data_dir}/{k}', transform[k]) for k in categories
}
# 数据加载器
dataloaders = {
    k: DataLoader(image_folders[k], batch_size=256, shuffle=True, num_workers=32, pin_memory=True) for k in categories
}

# train，val，test 大小
data_sizes = {
    k: len(image_folders[k]) for k in categories
}
# 类别名称
names = image_folders['train'].classes

# 显示一组图片
def imshow(imgs, title=None):
    imgs = imgs.numpy().transpose((1, 2, 0))  # 类型转换，交换维度顺序
    means = np.mean([mean_nums])  # 均值
    stds = np.mean([std_nums])  # 标准差
    imgs = imgs * stds + means  # 复原
    imgs = np.clip(imgs, 0, 1)  # 将像素限制在0-1之间
    plt.imshow(imgs)
    if title:
        plt.title(title)
    plt.axis('off')

# 显示train中一批数据 1个batch_size
inputs, labels = next(iter(dataloaders['train']))
group_imgs = make_grid(inputs)
imshow(group_imgs, title=[names[i] for i in labels])

# 迁移学习
def create_model(n_classes):
    """获取预训练模型
    n_classes : 类别数量
    """
    model = models.resnet50(pretrained=True)
    # 冻结模型参数
    for param in model.parameters():
        param.requires_grad = False
    # 替换全连接层
    model.fc = nn.Sequential(
        nn.Flatten(),  # 拉平
        nn.BatchNorm1d(2048),  # 正则化
        nn.Dropout(0.5),
        nn.Linear(2048, 512),
        nn.ReLU(),  # 激活函数
        nn.BatchNorm1d(512),
        nn.Dropout(0.5),
        nn.Linear(512, 43),
        nn.LogSoftmax(dim=1)
    )
    return model

# 训练函数
def train(model, data_loader, criterion, optimizer, scheduler, n_examples, device):
    model.train()
    optimizer.zero_grad()  # 梯度置零
    losses = []
    correct_predictions = 0
    for inputs, targets in data_loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        outputs = model(inputs)  # 输出
        loss = criterion(outputs, targets)  # 计算损失
        _, preds = torch.max(outputs, dim=1)  # 获取最大概率值的下标
        correct_predictions += torch.sum(preds == targets)  # 累计判断正确的数量
        losses.append(loss.item())  # 累计损失
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数、
    scheduler.step()  # 学习率自动调整
    return np.mean(losses), correct_predictions / n_examples

# 验证函数
def val(model, data_loader, criterion, n_examples, device):
    model.eval()
    losses = []
    correct_predictions = 0
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)  # 预测输出
            loss = criterion(outputs, targets)  # 计算损失
            _, preds = torch.max(outputs, dim=1)  # 获取最大概率值下标
            correct_predictions += torch.sum(preds == targets)  # 累计判断正确的数量
            losses.append(loss.item())  # 累计损失
    return np.mean(losses), correct_predictions / n_examples

def train_val_model(model, data_loader, dataset_sizes, device, epochs=10):
    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)  # 自动调节学习率
    criterion = nn.CrossEntropyLoss()  # 损失函数
    history = defaultdict(list)  # 保存结果
    best_accuracy = 0  # 最好的准确率
    best_model = None  # 最好的模型
    for epoch in range(epochs):
        print(f'{epoch + 1} / {epochs}')
        train_loss, train_accuracy = train(model, dataloaders['train'], criterion, optimizer,
                                           scheduler, data_sizes['train'], device)
        print(f'Train Loss : {train_loss}, Train Acc : {train_accuracy}')

        eval_loss, eval_accuracy = val(model, dataloaders['val'], criterion, data_sizes['val'], device)
        print(f'Eval Loss : {eval_loss}, Eval Acc : {eval_accuracy}')

        # 保存结果
        history['train_acc'].append(train_accuracy)
        history['train_loss'].append(train_loss)
        history['eval_acc'].append(eval_accuracy)
        history['eval_loss'].append(eval_loss)

        # 比较：获取最好得分和模型
        if best_accuracy < eval_accuracy:
            torch.save(model.state_dict(), 'best_model_state.pth')
            best_accuracy = eval_accuracy
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epochs':epochs
            }

    print("最好模型验证得分： ", best_accuracy)

    # 加载模型
    model.load_state_dict(torch.load("best_model_state.pth"))

    return model, history


device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #cuda:0代表起始的
# #device_id为0,如果直接是cuda,同样默认是从0开始，可以根据实际需要修改起始位置，如cuda:1
new_model = create_model(len(names)).to(device)  # 新模型

if torch.cuda.device_count() > 1: # 查看当前电脑的可用的gpu的数量，若gpu数量>1,就多gpu训练
    new_model = torch.nn.DataParallel(new_model) # gpu训练,自动选择gpu
new_model = new_model.to(device) # 将网络模型放到指定的gpu或cpu上

best_model, history = train_val_model(new_model, dataloaders, data_sizes, device, epochs=30)

# 根据预测结果，计算统计指标
def get_predictions(model, data_loaders):
    model.eval()
    predictions = [] # 预测值
    real_values = [] #真实值
    with torch.no_grad():
        for inputs, labels in data_loaders:
            inputs = inputs.to(device)
            labels = labels.to(device)
            # 预测输出
            outputs = model(inputs)
            # 获取概率最大值的索引
            _,preds = torch.max(outputs, dim=1)
            # 保存预测值和真实值
            predictions.extend(preds)
            real_values.extend(labels)
        print("predictions = ", predictions)
        print("real_values = ", real_values)
        # 类型转换
        predictions = torch.as_tensor(predictions).cpu()
        real_values = torch.as_tensor(real_values).cpu()

        return predictions, real_values

# 函数：对图片进行识别，计算各个类别的概率
def predict_proba(model, img_path):
    # 读取图片
    image = Image.open(img_path)
    image = transform['test'](image).unsqueeze(0) # 图像变化，并扩充一维，充当batch_size
    # 模型预测
    pred = model(image.to(device))
    print("output : ", pred)
    # 计算概率
    proba = F.softmax(pred, dim=1)
    print("proba : ", proba )

    # print("proba.data() ", proba.data()) 报错
    print("proba.detach()", proba.detach())
    return proba.detach().cpu().numpy().flatten() # flatten() : 返回一个一维数组

# 绘制loss, acc
def plot_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    ax1.plot(history['train_loss'], label="train loss")
    ax1.plot(history['eval_loss'], label="eval loss")
    # ax1.set_ylim([-0.05, 1.05])
    ax1.legend()
    ax1.set_ylabel("Loss")
    ax1.set_xlabel("Epoch")

    ax2.plot(history['train_acc'], label="train acc")
    ax2.plot(history['eval_acc'], label="eval acc")
    # ax2.set_ylim([-0.05, 1.05])
    ax2.legend()
    ax2.set_ylabel("Accuracy")
    ax2.set_xlabel("Epoch")

    fig.suptitle("Train and Val history")