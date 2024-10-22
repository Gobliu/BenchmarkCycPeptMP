import torch
from tqdm import tqdm

from DecayCosineAnnealingWarmRestarts import DecayCosineAnnealingWarmRestarts

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class ModelHandler:
    def __init__(self, model, lr, loss, sch_step):
        self.model = model
        self.opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.loss = loss
        self.sch = DecayCosineAnnealingWarmRestarts(self.opt, sch_step, [0.9], 0.001*lr)

    def train(self, data_loader):
        loss_list = []

        self.model.train()
        for _, (sample_id, x, y, w, _) in enumerate(tqdm(data_loader)):
            self.opt.zero_grad()
            # print(x)
            x = x.to(device)
            y = y.to(device)
            pred = self.model(x)
            # print('pred', pred, pred.shape, 'y', y, y.shape)
            loss = self.loss(pred, y)
            loss = torch.mean(loss * w)
            loss.backward()
            self.opt.step()
            loss_list.append(loss.item())

        mean_loss = torch.mean(torch.tensor(loss_list))
        return mean_loss

    def eval(self, data_loader):
        loss_list = []

        self.model.eval()
        with torch.no_grad():
            for _, (sample_id, x, y, w, _) in enumerate(tqdm(data_loader)):
                x = x.to(device)
                y = y.to(device)
                pred = self.model(x)
                loss = self.loss(pred, y)
                loss = torch.mean(loss * w)
                loss_list.append(loss.item())

        mean_loss = torch.mean(torch.tensor(loss_list))
        return mean_loss

    def inference(self, x):
        self.model.eval()
        with torch.no_grad():
            x = x.to(device)
            pred = self.model(x)
        return pred
