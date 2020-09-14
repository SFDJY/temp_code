import os

import torch
from torch import nn, optim
from torch.autograd import Variable

from sfdjy_dataset import sfdjy_data_loader
from SFDJY_BiSeNet import BiSeNet
from sfdjy_utils import Evaluator
from sfdjy_dataset import n_class


def sfdjy_train(model, epochs=100):
    for epoch in range(epochs):
        model.train()
        for i, data in enumerate(sfdjy_data_loader):
            image, label = data
            image = image.permute(0, 3, 1, 2)
            label = label.long()
            image = image.cuda()
            label = label.cuda()
            image = Variable(image)
            label = Variable(label)

            out = model(image)

            criterion = nn.CrossEntropyLoss()
            optimizer = optim.SGD(model.parameters(), lr=0.001)

            loss = criterion(out, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _, pred = torch.max(out, dim=1)

            acc = (pred == label).float().mean()

            if i % 100 == 0:
                print('=============================== ' + str(i) + ' ===================================')
                print("epoch: {}/{}, loss: {:.6f}, running_acc: {:.6f}"
                      .format(epoch + 1, epochs, loss.item(), acc.item()))

#         model.eval()
#         accuracy = 0.0
#         number_train_data = 0
#         sfdjy_miou = Evaluator(num_class=n_class)
#         for i, data in enumerate(sfdjy_test_data_loader):
#             image, label = data
#             image = image.permute(0, 3, 1, 2)
#             label = label.long()
#             image = image.cuda()
#             label = label.cuda()
#             image = Variable(image)
#             label = Variable(label)

#             out = model(image)

#             _, pred = torch.max(out, dim=1)

#             accuracy += (pred == label).float().mean()
#             print('acc = ', (pred == label).float().mean())

#             sfdjy_miou.add_batch(label.cpu(), pred.cpu())
#             print('miou = ', sfdjy_miou.Mean_Intersection_over_Union())
#             sfdjy_miou.reset()
#             number_train_data = i

#         print('accuracy = ', accuracy / number_train_data)

        torch.save(model.state_dict(), './sfdjy_bisenet.pth')


if __name__ == "__main__":
    model = BiSeNet()
    if os.path.exists('./sfdjy_bisenet.pth'):
        model.load_state_dict(torch.load('./sfdjy_bisenet.pth'))
    model = model.cuda()
    sfdjy_train(model)
