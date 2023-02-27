from torch.utils.data import DataLoader
import argparse
from utils.datautil import *
import torch.nn as nn
import torch.nn.functional as F
import os


parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default='vae', help="name of model")
parser.add_argument("--epochs", type=int, default=50, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=5, help="size of the batches")
parser.add_argument("--sample_interval", type=int, default=200, help="interval between saving generator outputs")
parser.add_argument("--lr", type=float, default=0.0003, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.9, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--train_rate", type=float, default=0.8, help="rate of train data")
parser.add_argument("--sample_len", type=int, default=2000, help="length of signal")
parser.add_argument("--lambda_z", type=float, default=0.05, help="loss weight of z")
parser.add_argument("--z_dimension", type=int, default=256, help="dim of latent space")
parser.add_argument("--dataset", type=str, default="nature_data", help="name of the dataset")
parser.add_argument("--mode", type=str, default='train', help="name of model")


opt = parser.parse_args()
print(opt)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ecgs, imf5s, imf6s = load_data(opt)
ecgs, win_amplitude_datas, win_phase_datas = load_data(opt)

train_data, test_data = trainTestSplit(ecgs, win_amplitude_datas, win_phase_datas, opt.train_rate)

train_dataset = UWBECGDataSet(train_data)
test_dataset = UWBECGDataSet(test_data)

train_dataloader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)


def sample_signals(epoch, batch):
    os.makedirs('images/' + opt.model_name + '/train', exist_ok=True)

    data = next(iter(test_dataloader))
    vae.eval()

    ecg = data[0].type(torch.FloatTensor).to(device)
    amplitude_data = data[1].type(torch.FloatTensor).to(device)
    win_phase_datas = data[2].type(torch.FloatTensor).to(device)
    # input = torch.cat((imf5, imf6), 1)
    generated_ECG = vae(amplitude_data)

    plt.subplot(3, 1, 1)
    plt.plot(ecg[0, 0, :].cpu())
    plt.title('ECG')
    plt.subplot(3, 1, 2)
    plt.plot(amplitude_data[0, 0, :].cpu())
    plt.title('imf6')
    plt.subplot(3, 1, 3)
    plt.plot(generated_ECG[0][0, 0, :].cpu().detach().numpy())
    plt.title('generate FECG')

    plt.title('generate ECG')
    plt.savefig("images/%s/train/%d_%d.png" % (opt.model_name, epoch, batch))
    plt.close()


class VAE(nn.Module):
    def __init__(self, opt):
        super(VAE, self).__init__()
        # 定义编码器
        self.encoder = nn.Sequential(
            nn.Conv1d(1,4,kernel_size=5,stride=4,padding=1),
            nn.BatchNorm1d(4),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Conv1d(4,8,kernel_size=5,stride=4,padding=1),
            nn.BatchNorm1d(8),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Conv1d(8,16,kernel_size=5,stride=4,padding=1),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Conv1d(16, 32, kernel_size=5, stride=4, padding=1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.encoder_fc1=nn.Linear(32*1*8,opt.z_dimension)
        self.encoder_fc2=nn.Linear(32*1*8,opt.z_dimension)
        self.Sigmoid = nn.Sigmoid()
        self.decoder_fc = nn.Linear(opt.z_dimension,32*1*8)
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(32, 16, kernel_size=5, stride=4, padding=1),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose1d(16, 8, kernel_size=5, stride=4, padding=1, output_padding=2),
            nn.BatchNorm1d(8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose1d(8, 4, kernel_size=5, stride=4, padding=1, output_padding=1),
            nn.BatchNorm1d(4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose1d(4, 1, kernel_size=5, stride=4, padding=1, output_padding=1),
            nn.BatchNorm1d(1),
            # nn.Sigmoid(),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def noise_reparameterize(self,mean,logvar):
        eps = torch.randn(mean.shape).to(device)
        z = mean + eps * torch.exp(logvar)
        return z

    def forward(self, x):
        out1,out2 = self.encoder(x),self.encoder(x)
        mean = self.encoder_fc1(out1.view(out1.shape[0],-1))
        logstd = self.encoder_fc2(out2.view(out2.shape[0],-1))
        z = self.noise_reparameterize(mean,logstd)
        out3 = self.decoder_fc(z)
        out3 = out3.view(out3.shape[0], 32, 8)
        out3 = self.decoder(out3)
        return out3,mean,logstd

def loss_function(recon_x,x,mean,std):
    # BCE = F.binary_cross_entropy(recon_x,x,reduction='sum')
    BCE = F.l1_loss(recon_x, x)
    # 因为var是标准差的自然对数，先求自然对数然后平方转换成方差
    var = torch.pow(torch.exp(std),2)
    KLD = -0.5 * torch.sum(1+torch.log(var)-torch.pow(mean,2)-var)
    return BCE+opt.lambda_z * KLD


vae = VAE(opt).to(device)

optimizer = torch.optim.Adam(vae.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

losses = []

for epoch in range(0, opt.epochs):
    for i, data in enumerate(train_dataloader):
        ecg = data[0].type(torch.FloatTensor).to(device)
        amplitude_data = data[1].type(torch.FloatTensor).to(device)
        phase_data = data[2].type(torch.FloatTensor).to(device)

        x, mean, std = vae(amplitude_data)  # 将真实图片放入判别器中
        loss = loss_function(x, ecg, mean, std)
        optimizer.zero_grad()  # 在反向传播之前，先将梯度归0
        loss.backward()  # 将误差反向传播
        optimizer.step()  # 更新参数

        if (i + 1) % 50 == 0:
            print('Epoch[{}/{}],Batch[{}/{}], vae_loss:{:.6f} '.format(
                epoch+1, opt.epochs, i, len(train_dataloader),loss.item(),
            ))
            losses.append(loss.item())

        # If at sample interval save image
        if i % opt.sample_interval == 0:
            sample_signals(epoch, i)

    torch.save(vae.state_dict(), f"weights/vae.pth")

plt.plot(losses)
plt.title('loss')
plt.savefig("result/loss.png")
plt.close()