from torch.utils.data import DataLoader
import argparse
from utils.datautil import *
import torch.nn as nn
import torch.nn.functional as F
import os

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default='vae_gan', help="name of model")
parser.add_argument("--epochs", type=int, default=50, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=16, help="size of the batches")
parser.add_argument("--sample_interval", type=int, default=20, help="interval between saving generator outputs")
parser.add_argument("--lr", type=float, default=0.0003, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--train_rate", type=float, default=0.8, help="rate of train data")
parser.add_argument("--sample_len", type=int, default=2000, help="length of signal")
parser.add_argument("--lambda_z", type=float, default=0.5, help="loss weight of z")
parser.add_argument("--lambda_gan", type=float, default=1, help="loss weight of gan")
parser.add_argument("--z_dimension", type=int, default=512, help="dim of latent space")
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
    phase_data = data[2].type(torch.FloatTensor).to(device)

    input = amplitude_data

    generated_ECG = vae(input)

    plt.subplot(3, 1, 1)
    plt.plot(ecg[0, 0, :].cpu())
    plt.title('ECG')
    plt.subplot(3, 1, 2)
    plt.plot(input[0, 0, :].cpu())
    plt.title('imf6')
    plt.subplot(3, 1, 3)
    plt.plot(generated_ECG[0][0, 0, :].cpu().detach().numpy())
    plt.title('generate FECG')

    plt.title('generate ECG')
    plt.savefig("images/%s/train/%d_%d.png" % (opt.model_name, epoch, batch))
    plt.close()

class Discriminator(nn.Module):
    def __init__(self, opt):
        super(Discriminator, self).__init__()
        self.dis = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=4, kernel_size=31, stride=2),
            nn.BatchNorm1d(4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(in_channels=4, out_channels=8, kernel_size=5, stride=2),
            nn.BatchNorm1d(8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(in_channels=8, out_channels=16, kernel_size=5, stride=2),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, stride=2),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # self.fc = nn.Sequential(
        #     nn.Linear(97, 1),
        #     # nn.LeakyReLU(0.2, inplace=True),
        #     # nn.Linear(32, 1),
        #     nn.Sigmoid()
        # )
        hidden_size = 16
        num_layers = 1
        self.rnn = nn.LSTM(input_size=32,
                           hidden_size=hidden_size,
                           num_layers=num_layers,
                           batch_first=True,
                           bidirectional=True
                           )
        self.fc_rnn = nn.Linear(hidden_size*2, 1)
        self.m = nn.Sigmoid()

        # self.conv1 = nn.Conv1d(in_channels=1, out_channels=3, kernel_size=31, stride=2)
        # self.norm1 = nn.BatchNorm1d(3)
        # self.conv2 = nn.Conv1d(in_channels=3, out_channels=6, kernel_size=31, stride=2)
        # self.norm2 = nn.BatchNorm1d(6)
        # self.conv3 = nn.Conv1d(in_channels=6, out_channels=3, kernel_size=31, stride=2)
        # self.norm3 = nn.BatchNorm1d(3)
        # self.conv4 = nn.Conv1d(in_channels=3, out_channels=1, kernel_size=31, stride=2)
        # self.norm4 = nn.BatchNorm1d(1)
        #
        # self.fc1 = nn.Linear(122, 32)
        # self.fc2 = nn.Linear(32, 1)
        # self.m = nn.Sigmoid()
    # plt.plot(input[1,0,:].cpu().detach().numpy())
    def forward(self, input, hidden=None):
        x = self.dis(input)
        # x = x.view(x.size(0), -1)
        # x = self.fc(x)
        # x = x.unsqueeze(0)
        x = torch.transpose(x, 1, 2)
        output, hidden = self.rnn(x, hidden)
        x = output[:, -1, :]
        x = self.fc_rnn(x)
        x = self.m(x)
        # x = self.norm1(torch.sin(self.conv1(input)))
        # x = self.norm2(torch.sin(self.conv2(x)))
        # x = self.norm3(torch.sin(self.conv3(x)))
        # x = self.norm4(torch.sin(self.conv4(x)))
        # x = x.view(x.size(0), -1)
        # x = torch.sin(self.fc1(x))
        # x = self.m(self.fc2(x))
        return x.squeeze(1)


class VAE(nn.Module):
    def __init__(self, opt):
        super(VAE, self).__init__()
        # 定义编码器
        self.encoder = nn.Sequential(
            nn.Conv1d(1,4,kernel_size=31,stride=2),
            nn.BatchNorm1d(4),
            nn.LeakyReLU(0.2,),
            nn.Conv1d(4,8,kernel_size=5,stride=2,),
            nn.BatchNorm1d(8),
            nn.LeakyReLU(0.2,),
            nn.Conv1d(8,16,kernel_size=5,stride=2,),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.2,),
            nn.Conv1d(16, 32, kernel_size=5, stride=2,),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2, ),
        )
        self.encoder_fc1=nn.Linear(32*1*120,opt.z_dimension)
        self.encoder_fc2=nn.Linear(32*1*120,opt.z_dimension)
        self.Sigmoid = nn.Sigmoid()
        self.decoder_fc = nn.Linear(opt.z_dimension,32*1*120)
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(32, 16, kernel_size=5, stride=2, output_padding=1),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.2, ),
            nn.ConvTranspose1d(16, 8, kernel_size=5, stride=2, ),
            nn.BatchNorm1d(8),
            nn.LeakyReLU(0.2, ),
            nn.ConvTranspose1d(8, 4, kernel_size=5, stride=2, ),
            nn.BatchNorm1d(4),
            nn.LeakyReLU(0.2, ),
            nn.ConvTranspose1d(4, 1, kernel_size=31, stride=2, output_padding=1),
            # nn.Tanh()
            # nn.BatchNorm1d(1),
            nn.LeakyReLU(0.2, ),
        )

        # self.conv1 = nn.Conv1d(1, 4, kernel_size=31, stride=2, )
        # self.norm1 = nn.BatchNorm1d(4)
        # self.conv2 = nn.Conv1d(4, 8, kernel_size=5, stride=2, )
        # self.norm2 = nn.BatchNorm1d(8)
        # self.conv3 = nn.Conv1d(8, 16, kernel_size=5, stride=2, )
        # self.norm3 = nn.BatchNorm1d(16)
        # self.conv4 = nn.Conv1d(16, 32, kernel_size=5, stride=2, )
        # self.norm4 = nn.BatchNorm1d(32)

        # self.convt1 = nn.ConvTranspose1d(32, 16, kernel_size=5, stride=2, output_padding=1)
        # self.normt1 = nn.BatchNorm1d(16)
        # self.convt2 = nn.ConvTranspose1d(16, 8, kernel_size=5, stride=2, )
        # self.normt2 = nn.BatchNorm1d(8)
        # self.convt3 = nn.ConvTranspose1d(8, 4, kernel_size=5, stride=2,)
        # self.normt3 = nn.BatchNorm1d(4)
        # self.convt4 = nn.ConvTranspose1d(4, 1, kernel_size=31, stride=2, output_padding=1)
        # self.normt4 = nn.BatchNorm1d(1)
    #
    # def encoder(self,input):
    #     x = self.norm1(torch.sin(self.conv1(input)))
    #     x = self.norm2(torch.sin(self.conv2(x)))
    #     x = self.norm3(torch.sin(self.conv3(x)))
    #     x = self.norm4(torch.sin(self.conv4(x)))
    #     return x
    #
    # def decoder(self, x):
    #     x = self.normt1(torch.sin(self.convt1(x)))
    #     x = self.normt2(torch.sin(self.convt2(x)))
    #     x = self.normt3(torch.sin(self.convt3(x)))
    #     # x = self.normt4(torch.sin(self.convt4(x)))
    #     x = torch.sin(self.convt4(x))
    #     return x

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
        out3 = out3.view(out3.shape[0], 32, 120)
        out3 = self.decoder(out3)
        return out3,mean,logstd


def loss_function(recon_x,x,mean,std):
    # BCE = F.binary_cross_entropy(recon_x,x,reduction='sum')
    BCE = F.mse_loss(recon_x, x)
    # 因为var是标准差的自然对数，先求自然对数然后平方转换成方差
    var = torch.pow(torch.exp(std),2)
    KLD = -0.5 * torch.sum(1+torch.log(var)-torch.pow(mean,2)-var)
    return BCE + opt.lambda_z * KLD


vae = VAE(opt).to(device)
D = Discriminator(opt).to(device)

optimizer = torch.optim.Adam(vae.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizerD = torch.optim.Adam(D.parameters(), lr=0.0001, betas=(opt.b1, opt.b2))

torch.autograd.set_detect_anomaly(True)

MSECriterion = nn.MSELoss().to(device)
BCECriterion = nn.BCELoss().to(device)
losses = []

for epoch in range(0, opt.epochs):
    for i, data in enumerate(train_dataloader):
        ecg = data[0].type(torch.FloatTensor).to(device)
        amplitude_data = data[1].type(torch.FloatTensor).to(device)
        phase_data = data[2].type(torch.FloatTensor).to(device)

        input = amplitude_data

        ###################################################################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###################################################################
        # train with real
        D.zero_grad()
        output = D(ecg)
        real_label = torch.ones(ecg.shape[0]).to(device)  # 定义真实的图片label为1
        fake_label = torch.zeros(ecg.shape[0]).to(device)  # 定义假的图片的label为0
        errD_real = BCECriterion(output, real_label)
        errD_real.backward()
        real_data_score = output.mean().item()
        # 随机产生一个潜在变量，然后通过decoder 产生生成图片
        z = torch.randn(ecg.shape[0], opt.z_dimension).to(device)
        # 通过vae的decoder把潜在变量z变成虚假图片
        fake_data = vae.decoder_fc(z).view(z.shape[0], 32, 120)
        fake_data = vae.decoder(fake_data)
        output = D(fake_data)
        errD_fake = BCECriterion(output, fake_label)
        errD_fake.backward()
        # fake_data_score用来输出查看的，是虚假照片的评分，0最假，1为真
        fake_data_score = output.data.mean()
        errD = errD_real + errD_fake
        optimizerD.step()

        ###################################################
        # (2) Update G network which is the decoder of VAE
        ###################################################
        recon_data, mean, logstd = vae(input)
        vae.zero_grad()
        vae_loss = loss_function(recon_data, input, mean, logstd)
        vae_loss.backward()
        optimizer.step()

        ###############################################
        # (3) Update G network: maximize log(D(G(z)))
        ###############################################
        vae.zero_grad()
        real_label = torch.ones(ecg.shape[0]).to(device)  # 定义真实的图片label为1
        recon_data, mean, logstd = vae(input)
        output = D(recon_data)
        errVAE = BCECriterion(output, real_label) * opt.lambda_gan
        errVAE.backward()
        D_G_z2 = output.mean().item()
        optimizer.step()


        if (i + 1) % 5 == 0:
            print('Epoch[%d/%d],Batch[%d/%d], r_score: %.4f f_score: %.4f vae_loss: %.4f' % (
                epoch+1, opt.epochs, i, len(train_dataloader),
                real_data_score,
                fake_data_score,
                vae_loss.item(),
            ))
            losses.append(vae_loss.item())

        # If at sample interval save image
        if i % opt.sample_interval == 0:
            sample_signals(epoch, i)

    torch.save(vae.state_dict(), f"weights/vae+_gan.pth")

plt.plot(losses)
plt.title('loss')
plt.savefig("result/loss.png")
plt.close()