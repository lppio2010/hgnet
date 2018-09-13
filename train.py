import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from fer2013_dataset import Fer2013Dataset, Fer2013ValidDataset
from backbone import resnet

# for param in model.parameters():
#     print(type(param.data), param.size())
# for idx, m in enumerate(model.modules()):
#     print(idx, '->', m)
# for name, param in model.named_parameters():
#     print(name, param.size())

def train(train_loader, model, criterion, optimizer, epoch, device, num_epochs, log_interval):
    batch_time = AverageMeter()
    data_time = AverageMeter()

    model.train()
    train_len = len(train_loader)
    epoch_acc = 0
    end = time.time()

    for batch_idx, batch_sample in enumerate(train_loader):
        data, target = batch_sample['image'].to(device), batch_sample['label'].to(device)

        data_time.update(time.time() - end)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        acc = accuracy(output, target)
        
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        epoch_acc += acc
        if batch_idx % log_interval == 0:
            print('Train Epoch: {}\t[{}/{} ({:.0f}%)]\t'
                  'Loss: {:.6f}\tAccuracy: {:.6f}\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})'.format(
                  epoch, batch_idx * len(data), len(train_loader.dataset),
                  100. * batch_idx / len(train_loader),
                  loss.item(), acc,
                  batch_time=batch_time, data_time=data_time))
    return epoch_acc

def test(test_loader, model, criterion, device):
    model.eval()
    test_loss = 0
    acc = 0
    test_len = len(test_loader)
    with torch.no_grad():
        for batch_sample in test_loader:
            data, target = batch_sample['image'].to(device), batch_sample['label'].to(device)
            output = model(data)
            test_loss += criterion(output, target.view(target.size(0))).item()
            acc += accuracy(output, target, mode=0)

    test_loss /= test_len
    test_acc = acc / test_len
    print('\nTest set: Average loss: {:.6f}, Accuracy: {:.6f} \n'.format(
        test_loss, test_acc))
    return test_acc

def accuracy(output, target, mode=0):
    predict_idx = torch.argmax(output, dim=1)
    total_correct = torch.eq(target, predict_idx).sum().item()
    if mode == 1:
        print(total_correct)
    batch_size = target.size(0)
    return total_correct / batch_size

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def main():
    seed = 11451466
    num_class = 7
    lr = 0.01
    momentum = 0.9
    weight_decay = 0.0001
    test_intvl = 1
    num_epochs = 100
    cuda = torch.cuda.is_available()

    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)

    device = torch.device("cuda" if cuda else "cpu")

    img_transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToPILImage(),
        torchvision.transforms.ToTensor()
    ])

    train_loader = torch.utils.data.DataLoader(Fer2013Dataset(transforms=img_transforms), batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(Fer2013ValidDataset(transforms=img_transforms), batch_size=64)

    model = resnet.resnet50()
    model.to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    # scheduler = nn.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=1)

    # for param in model.parameters():
    #     print(type(param.data), param.size())
    # for idx, m in enumerate(model.modules()):
    #     print(idx, '->', m)
    # for name, param in model.named_parameters():
    #     print(name, param.size())

    log_interval = len(train_loader)//4
    best_acc = 0
    for epoch in range(1, num_epochs + 1):
        acc = train(train_loader, model, criterion, optimizer, epoch, device, num_epochs, log_interval)
        acc /= len(train_loader)
        # scheduler.step(acc)
        if epoch % test_intvl == 0:
            best_acc = max(best_acc, test(test_loader, model, criterion, device))
    best_acc = max(best_acc, test(test_loader, model, criterion, device))
    print('best test accuracy: {:.6f}'.format(best_acc))

    # snapshot(model, snapshot_folder, num_epochs)

if __name__ == '__main__':
    main()