import torch
from tqdm import tqdm

from DecayCosineAnnealingWarmRestarts import DecayCosineAnnealingWarmRestarts

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class ModelHandler:
    def __init__(self, model, lr, loss, sch_step):
        self.model = model
        self.opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.loss = loss
        print('loss function', loss)
        self.sch = DecayCosineAnnealingWarmRestarts(self.opt, sch_step, [0.8], 0.0001*lr)

    def train(self, data_loader):
        loss_list = []

        self.model.train()
        for _, (sample_id, x, y, w, _) in enumerate(tqdm(data_loader)):
            self.opt.zero_grad()
            # print(x[:2, :10]) 
            x = x.to(device)
            y = y.to(device)
            pred = self.model(x)
            # print('pred', pred.flatten())
            # print('pred', pred.dtype, 'y', y.dtype)
            loss = self.loss(pred, y.float())
            # print('pred', pred.flatten(), 'loss', loss.flatten())
            loss = torch.mean(loss * w)
            loss.backward()
            self.opt.step()
            loss_list.append(loss.item())

        mean_loss = torch.mean(torch.tensor(loss_list))
        return mean_loss

    def eval(self, data_loader):
        loss_list = []

        self.model.eval()
        with torch.inference_mode():
            for _, (sample_id, x, y, w, _) in enumerate(tqdm(data_loader)):
                x = x.to(device)
                y = y.to(device)
                pred = self.model(x)
                loss = self.loss(pred, y.float())
                loss = torch.mean(loss * w)
                loss_list.append(loss.item())

        mean_loss = torch.mean(torch.tensor(loss_list))
        return mean_loss

    def inference(self, x):
        self.model.eval()
        with torch.inference_mode():
            x = x.to(device)
            pred = self.model(x)
        return pred
