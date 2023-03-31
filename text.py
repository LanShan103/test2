import matplotlib.pyplot as plt
import numpy as np
import torchvision
import torchvision.transforms as transforms
from Demos.BackupRead_BackupWrite import outfile
from torchvision import datasets

#train_image = Image.open(r"E:\clip\实验二\root\img.png")
#val_image = Image.open(r'E:\clip\实验二\root\img_1.png')

train_dataset = datasets.ImageFolder(root='E:\\clip\\实验二\\root\\train',
            transform=transforms.Compose([
                transforms.RandomResizedCrop(256),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.RandomResizedCrop(size=256, scale=(0.7, 1.0), ratio=(3. / 4., 4. / 3.), interpolation=2),
                transforms.ToTensor()
                # transforms.Normalize(opt.data_mean, opt.data_std)
            ])
                                     )
val_dataset = datasets.ImageFolder(root='E:\\clip\\实验二\\root\\val',
    transform=transforms.Compose([
        transforms.RandomResizedCrop(256),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.RandomResizedCrop(size=256, scale=(0.7, 1.0), ratio=(3. / 4., 4. / 3.), interpolation=2),
        transforms.ToTensor()
        # transforms.Normalize(opt.data_mean, opt.data_std)
    ])
)
import matplotlib.pyplot as plt
import numpy as np

# 定义一个函数用于可视化图像数据
def imshow(img):
    img = img / 2 + 0.5 # 反标准化操作
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))

# 随机获取一些训练数据并进行可视化
dataiter = iter(train_dataset)
images, labels = dataiter.__next__()
imshow(torchvision.utils.make_grid(images))
plt.show()  # 显示图像

dataiter = iter(val_dataset)
images, labels = dataiter.__next__()
imshow(torchvision.utils.make_grid(images))
plt.show()  # 显示图像