import torch
from torchvision import datasets, transforms


def load_mnist(batch_size, test_batch_size, use_cuda, path):
    """load the mnist from the path if existing
    or from the Internet
    
    Arguments:
        batch_size {int} --easy to know
        test_batch_size {int} --easy to know
        use_cuda {bool} -- use the cuda if use_cude is true
        path {str} -- the path of the existing or downloading data
    
    Returns:
        data_loader -- 数据集的一个迭代器
    """
    print("Loading data...")
    kwargs = {'num_workers': 4, 'pin_memory': False} if use_cuda else {}
    train_dataset = datasets.MNIST(path,
                                    train=True,  # 是否加载数据库的训练集，false的时候加载测试集
                                    download=True,
                                    transform=transforms.Compose([
                                        transforms.ToTensor(),  # 将像素值范围[0,255]转换到[0.0,1.0]
                                        transforms.Normalize((0.1307,), (0.3081,))
                                    ]))
    test_dataset = datasets.MNIST(path,
                                    train=False,
                                    transform=transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.1307,), (0.3081,))
                                    ]))
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,  # 默认batch 1000
        shuffle=True,  # 是否随机打乱
        **kwargs
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=test_batch_size,
        shuffle=True,
        **kwargs
    )
    return train_loader, test_loader