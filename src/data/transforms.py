import torchvision.transforms as T

# 现在的 transforms.py 只有一个 cifar10_transform()，训练和测试用同一个。
# 问题：训练时没有随机性，模型容易过拟合，泛化差。

# 训练集做随机裁剪 + 随机翻转，验证/测试集只做归一化。 这是分类任务的标准做法。
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)


def cifar10_train_transform() -> T.Compose:
    """训练集：随机裁剪 + 随机翻转 + 归一化。"""
    return T.Compose(
        [
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )


def cifar10_test_transform() -> T.Compose:
    """验证/测试集：只做归一化，不做随机。"""
    return T.Compose(
        [
            T.ToTensor(),
            T.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )
