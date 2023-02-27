"""

"""
from torch.utils.data import DataLoader
import argparse
from utils.datautil import *
import torch.nn as nn
import os
from utils.evaluation_matrix import *
from torch.optim.lr_scheduler import StepLR


parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default='pix2pix_nature', help="name of model")
parser.add_argument("--epochs", type=int, default=100, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=32, help="size of the batches")
parser.add_argument("--sample_interval", type=int, default=50, help="interval between saving generator outputs")
parser.add_argument("--lr", type=float, default=0.001, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--leaky_relu_param", type=float, default=0.2, help="leaky_relu_param between 0.01 to 0.1-0.2")
parser.add_argument("--sample_len", type=int, default=2000, help="length of signal")
parser.add_argument("--lambda_pixel", type=float, default=5, help="loss weight of lambda_pixel")
parser.add_argument("--dataset", type=str, default="nature_data", help="nature_data or our_uwb or uwb200Hz")
parser.add_argument("--mode", type=str, default='train', help="train train eval_nature")

opt = parser.parse_args()
print(opt)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
# ecgs, win_amplitude_datas, win_phase_datas = load_data(opt)

app = ''
train_dataset_file = 'data/train_nature%s.npy' % (app)
test_dataset_file = 'data/test_nature%s.npy' % (app)

if opt.mode == 'train':
    train_data = np.load(train_dataset_file)
    test_data = np.load(test_dataset_file)
    train_dataset = UWBECGDataSet(train_data)
    train_dataloader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True)
    test_dataset = UWBECGDataSet(test_data)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)
elif opt.mode == 'test':
    test_data = np.load(test_dataset_file)
    test_dataset = UWBECGDataSet(test_data)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)


def sample_signals(epoch, batch):
    os.makedirs('images/' + opt.model_name + '/train', exist_ok=True)

    data = next(iter(test_dataloader))
    G.eval()

    ecg = data[0].type(torch.FloatTensor).to(device)
    amplitude_data = data[1].type(torch.FloatTensor).to(device)
    phase_data = data[2].type(torch.FloatTensor).to(device)

    generated_ECG = G(amplitude_data, phase_data)

    fig = plt.figure()
    plt.subplot(4, 1, 1)
    plt.plot(ecg[0, 0, :].cpu())
    plt.title('ECG')
    plt.subplot(4, 1, 2)
    plt.plot(generated_ECG[0, 0, :].cpu().detach().numpy())
    plt.title('generate ECG')
    plt.subplot(4, 1, 3)
    plt.plot(amplitude_data[0, 0, :].cpu())
    plt.title('amplitude_data')
    plt.subplot(4, 1, 4)
    plt.plot(phase_data[0, 0, :].cpu())
    plt.title('phase')

    fig.tight_layout()
    plt.savefig("images/%s/train/%d_%d.png" % (opt.model_name, epoch, batch))
    plt.close()


class Discriminator(nn.Module):
    def __init__(self, opt):
        super(Discriminator, self).__init__()

        self.dis1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=4, kernel_size=31, stride=1),
            nn.InstanceNorm1d(opt.sample_len),
            nn.LeakyReLU(opt.leaky_relu_param, inplace=True),
            nn.Conv1d(in_channels=4, out_channels=8, kernel_size=21, stride=1),
            nn.InstanceNorm1d(opt.sample_len),
            nn.LeakyReLU(opt.leaky_relu_param, inplace=True),
            nn.Conv1d(in_channels=8, out_channels=4, kernel_size=9, stride=1),
            nn.InstanceNorm1d(opt.sample_len),
            nn.LeakyReLU(opt.leaky_relu_param, inplace=True),
            nn.Conv1d(in_channels=4, out_channels=1, kernel_size=5, stride=1),
            nn.InstanceNorm1d(opt.sample_len),
            nn.LeakyReLU(opt.leaky_relu_param, inplace=True),
        )
        self.dis2 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=4, kernel_size=31, stride=1),
            nn.InstanceNorm1d(opt.sample_len),
            nn.LeakyReLU(opt.leaky_relu_param, inplace=True),
            nn.Conv1d(in_channels=4, out_channels=8, kernel_size=21, stride=1),
            nn.InstanceNorm1d(opt.sample_len),
            nn.LeakyReLU(opt.leaky_relu_param, inplace=True),
            nn.Conv1d(in_channels=8, out_channels=4, kernel_size=9, stride=1),
            nn.InstanceNorm1d(opt.sample_len),
            nn.LeakyReLU(opt.leaky_relu_param, inplace=True),
            nn.Conv1d(in_channels=4, out_channels=1, kernel_size=5, stride=1),
            nn.InstanceNorm1d(opt.sample_len),
            nn.LeakyReLU(opt.leaky_relu_param, inplace=True),
        )
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=2, out_channels=1, kernel_size=5, stride=1),
            nn.InstanceNorm1d(opt.sample_len),
            nn.LeakyReLU(opt.leaky_relu_param, inplace=True),
        )
        self.fc = nn.Sequential(
            nn.Linear(1934, 64),
            nn.LeakyReLU(opt.leaky_relu_param, inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, ecg, amplitude, phase):
        x1 = self.dis1(amplitude)
        x2 = self.dis2(ecg)
        x = torch.cat((x1, x2), 1)
        x = self.conv1(x)
        x = self.fc(x)

        return x.squeeze(1).squeeze(1)


class Pix2Pix(nn.Module):
    def __init__(self, opt):
        super(Pix2Pix, self).__init__()

        pad31 = max(31 - 1, 0) // 2
        pad5 = max(5 - 1, 0) // 2
        pad21 = max(21 - 1, 0) // 2
        pad9 = max(9 - 1, 0) // 2

        # 定义编码器
        self.encoder1 = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=31, padding=pad31),
            nn.InstanceNorm1d(opt.sample_len),
            nn.LeakyReLU(opt.leaky_relu_param,),
            # nn.Dropout(0.3),
            nn.Conv1d(8, 16, kernel_size=31, padding=pad31),
            nn.InstanceNorm1d(opt.sample_len),
            nn.LeakyReLU(opt.leaky_relu_param,),
            # nn.Dropout(0.3),
            nn.Conv1d(16, 32, kernel_size=31, padding=pad31),
            nn.InstanceNorm1d(opt.sample_len),
            nn.LeakyReLU(opt.leaky_relu_param,),
            # nn.Dropout(0.3),
            nn.Conv1d(32, 64, kernel_size=31, padding=pad31),
            nn.InstanceNorm1d(opt.sample_len),
            nn.LeakyReLU(opt.leaky_relu_param, ),
            # nn.Dropout(0.3),
        )

        self.encoder2 = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=31, padding=pad31),
            nn.InstanceNorm1d(opt.sample_len),
            nn.LeakyReLU(opt.leaky_relu_param,),
            # nn.Dropout(0.3),
            nn.Conv1d(8, 16, kernel_size=31, padding=pad31),
            nn.InstanceNorm1d(opt.sample_len),
            nn.LeakyReLU(opt.leaky_relu_param,),
            # nn.Dropout(0.3),
            nn.Conv1d(16, 32, kernel_size=31, padding=pad31),
            nn.InstanceNorm1d(opt.sample_len),
            nn.LeakyReLU(opt.leaky_relu_param,),
            # nn.Dropout(0.3),
            nn.Conv1d(32, 64, kernel_size=31, padding=pad31),
            nn.InstanceNorm1d(opt.sample_len),
            nn.LeakyReLU(opt.leaky_relu_param, ),
            # nn.Dropout(0.3),
        )

        self.decoder = nn.Sequential(
            nn.Conv1d(128, 64, kernel_size=5, padding=pad5),
            nn.InstanceNorm1d(opt.sample_len),
            nn.LeakyReLU(opt.leaky_relu_param, ),
            # nn.Dropout(0.3),
            nn.Conv1d(64, 32, kernel_size=9, padding=pad9),
            nn.InstanceNorm1d(opt.sample_len),
            nn.LeakyReLU(opt.leaky_relu_param, ),
            # nn.Dropout(0.3),
            nn.Conv1d(32, 16, kernel_size=9, padding=max(9 - 1, 0) // 2),
            nn.InstanceNorm1d(opt.sample_len),
            nn.LeakyReLU(opt.leaky_relu_param, ),
            # nn.Dropout(0.3),
            nn.Conv1d(16, 8, kernel_size=9, padding=max(9 - 1, 0) // 2),
            nn.InstanceNorm1d(opt.sample_len),
            nn.LeakyReLU(opt.leaky_relu_param, ),
            # nn.Dropout(0.3),
            nn.Conv1d(8, 4, kernel_size=21, padding=max(21 - 1, 0) // 2),
            nn.InstanceNorm1d(opt.sample_len),
            nn.LeakyReLU(opt.leaky_relu_param, ),
            # nn.Dropout(0.3),
            nn.Conv1d(4, 1, kernel_size=31, padding=max(31 - 1, 0) // 2),
            nn.Tanh(),
        )

    def forward(self, amplitude, angle):
        x1 = self.encoder1(amplitude)
        x2 = self.encoder1(angle)
        x = torch.cat((x1, x2), 1)
        out = self.decoder(x)
        return out


G = Pix2Pix(opt).to(device)
D = Discriminator(opt).to(device)


def train_main():
    print("start to train")

    optimizerG = torch.optim.Adam(G.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizerD = torch.optim.Adam(D.parameters(), lr=0.0001, betas=(opt.b1, opt.b2))

    # schedulerG = StepLR(optimizer=optimizerG, step_size=5, gamma=0.8)
    # schedulerD = StepLR(optimizer=optimizerD, step_size=5, gamma=0.8)

    schedulerG = StepLR(optimizer=optimizerG, step_size=10, gamma=0.9)
    schedulerD = StepLR(optimizer=optimizerD, step_size=10, gamma=0.9)

    MSECriterion = nn.MSELoss().to(device)
    BCECriterion = nn.BCELoss().to(device)
    L1Loss = nn.L1Loss().to(device)

    loss_Ds = []
    loss_Gs = []
    loss_GANs = []
    loss_pixels = []

    for epoch in range(0, opt.epochs):
        for i, data in enumerate(train_dataloader):
            ecg = data[0].type(torch.FloatTensor).to(device)
            amplitude_data = data[1].type(torch.FloatTensor).to(device)
            phase_data = data[2].type(torch.FloatTensor).to(device)

            real_label = torch.ones(ecg.shape[0]).to(device)  # 定义真实的图片label为1
            fake_label = torch.zeros(ecg.shape[0]).to(device)  # 定义假的图片的label为0

            # (1) Update G network: Generators
            optimizerG.zero_grad()

            # GAN loss
            fake_ECG = G(amplitude_data, phase_data)
            pred_fake = D(fake_ECG, amplitude_data, phase_data)
            loss_GAN = MSECriterion(pred_fake, real_label)

            # Pixel-wise loss
            loss_pixel = L1Loss(fake_ECG, ecg)

            loss_G = loss_GAN + opt.lambda_pixel * loss_pixel

            loss_G.backward()
            optimizerG.step()
            # 準確率計算
            g_accuracy = sum((pred_fake > 0.5))/pred_fake.shape[0]

            ##############################################
            # (2) Update D network:
            ##############################################
            # Set D_A gradients to zero
            optimizerD.zero_grad()

            train_ecg = torch.cat((ecg, fake_ECG.detach()), 0)
            train_amplitude = torch.cat((amplitude_data, amplitude_data), 0)
            train_phase = torch.cat((phase_data, phase_data), 0)
            train_labels = torch.cat((real_label, fake_label), 0)
            pred_D = D(train_ecg, train_amplitude, train_phase)
            loss_D = MSECriterion(pred_D, train_labels)
            loss_D.backward()
            optimizerD.step()

            d_accuracy = sum(train_labels.cpu().detach().numpy() == np.int64(np.asarray(pred_D.cpu().detach().numpy()) > 0.5))/train_labels.shape[0]

            if (i + 1) % 5 == 0:
                print('E[%d/%d],B[%d/%d], [Dloss: %.2f Dacc: %2f][Gloss: %.2f, adv: %.2f, pix: %.2f, Gacc: %2f]' % (
                    epoch+1, opt.epochs, i, len(train_dataloader),
                    loss_D.item(),
                    d_accuracy,
                    loss_G.item(),
                    loss_GAN.item(),
                    loss_pixel.item(),
                    g_accuracy,
                ))
                loss_Ds.append(loss_D.item())
                loss_Gs.append(loss_G.item())
                loss_GANs.append(loss_GAN.item())
                loss_pixels.append(loss_pixel.item())

            # If at sample interval save image
            if i % opt.sample_interval == 0:
                sample_signals(epoch, i)
        schedulerG.step()
        schedulerD.step()
        print('G learn rate：', schedulerG.get_last_lr())
        print('D learn rate：', schedulerD.get_last_lr())
        if epoch > 10 and epoch % 2 == 1:
            torch.save(G.state_dict(), "weights/pix2pix_nature_%d.pth"%(epoch))

    plot_data = [loss_Ds, loss_Gs, loss_GANs, loss_pixels]
    plot_data_title = ["loss_Ds", "loss_Gs", "loss_GANs", "loss_pixels"]

    for data, title in zip(plot_data, plot_data_title):
        plt.plot(data)
        plt.title(title)
        plt.savefig("result/%s.png" % (title))
        plt.close()


def test_main(model_path):
    print('start to test:', model_path.split('/')[-1])
    os.makedirs('images/' + opt.model_name + '/test/pic', exist_ok=True)
    os.makedirs('images/' + opt.model_name + '/test/npy', exist_ok=True)

    G.load_state_dict(torch.load(model_path))
    G.eval()

    cc_scores = []
    cos_scores = []
    rmse_scores = []
    ecgs = []
    gen_ecgs = []
    for index, data in enumerate(test_dataloader):
        ecg = data[0].type(torch.FloatTensor).to(device)
        ecgs.append(data[0].detach().numpy()[0,0,:])
        amplitude_data = data[1].type(torch.FloatTensor).to(device)
        phase_data = data[2].type(torch.FloatTensor).to(device)

        generated_ECG = G(amplitude_data, phase_data)[0, 0, :].cpu().detach().numpy()
        gen_ecgs.append(generated_ECG)

        cc_score = cc(generated_ECG, ecg[0, 0, :].cpu().numpy())
        cc_scores.append(cc_score)
        cos_dis_score = cos_dis(generated_ECG, ecg[0, 0, :].cpu().numpy())
        cos_scores.append(cos_dis_score)
        rmse_score = rmse(generated_ECG, ecg[0, 0, :].cpu().numpy())
        rmse_scores.append(rmse_score)

        # time_x = np.linspace(0, 10, 2000)
        # fig = plt.figure()
        # plt.subplot(4, 1, 1)
        # plt.plot(time_x, ecg[0, 0, :].cpu())
        # plt.title('ECG')
        # plt.subplot(4, 1, 2)
        # plt.plot(time_x, generated_ECG)
        # plt.title('generated_ECG')
        # plt.subplot(4, 1, 3)
        # plt.plot(time_x, amplitude_data[0, 0, :].cpu())
        # plt.title('amplitude_data')
        # plt.subplot(4, 1, 4)
        # plt.plot(time_x, phase_data[0, 0, :].cpu())
        # plt.title('phase_data')
        # fig.tight_layout()
        # plt.savefig("images/%s/test/pic/%d.png" % (opt.model_name, index))
        # plt.close()

    np.save("images/%s/test/npy/ecgs.npy"% (opt.model_name), np.asarray(ecgs))
    np.save("images/%s/test/npy/gen_ecgs.npy" % (opt.model_name), np.asarray(gen_ecgs))

    print('end to test')
    print("===========", opt.dataset, "===========")

    cc_scores = np.asarray(cc_scores)
    print('mean cc score', np.mean(cc_scores))
    print('median cc score', np.mean(cc_scores))

    cos_scores = np.asarray(cos_scores)
    print('mean cos score', np.mean(cos_scores))
    print('median cos score', np.mean(cos_scores))

    rmse_scores = np.asarray(rmse_scores)
    print('mean rmse score', np.mean(rmse_scores))
    print('median rmse score', np.median(rmse_scores))

    np.save('images/temp/cos_scores_nature%s.npy' % (app), cos_scores)
    np.save('images/temp/rmse_scores_nature%s.npy' % (app), rmse_scores)
    np.save('images/temp/cc_scores_nature%s.npy' % (app), cc_scores)

    print('start to evaluate_ecg_interval')
    evaluate_ecg_interval('images/pix2pix_nature/test/npy/')
    print('end to evaluate_ecg_interval')


def eval_nature(model_path, data_path):
    print('start to eval')
    data_name = data_path.split('/')[-1][:-4]
    os.makedirs('images/temp/'+data_name, exist_ok=True)
    # load model
    G.load_state_dict(torch.load(model_path))
    G.eval()
    # load data
    data = sio.loadmat(data_path)
    file_ecg = data['ecg_data']
    file_amp_data = data['amplitude_data']
    file_pha_data = data['angle_data']

    gen_ecgs = []

    for index in range(file_ecg.shape[1]):
        # resample
        win_ecg = signal.resample(file_ecg[:, index], 2000, axis=0)
        win_amplitude = -signal.resample(file_amp_data[:, index], 2000, axis=0)
        win_phase = signal.resample(file_pha_data[:, index], 2000, axis=0)

        # z_score
        ecg = (win_ecg - np.mean(win_ecg)) / np.std(win_ecg)
        amplitude = (win_amplitude - np.mean(win_amplitude)) / np.std(win_amplitude)
        phase = (win_phase - np.mean(win_phase)) / np.std(win_phase)

        # map to [-1, 1]
        ecg = mapminmax(ecg)
        amplitude = mapminmax(amplitude)
        phase = mapminmax(phase)

        amplitude = torch.as_tensor(amplitude).type(torch.FloatTensor).unsqueeze(0).unsqueeze(0).to(device)
        phase = torch.as_tensor(phase).type(torch.FloatTensor).unsqueeze(0).unsqueeze(0).to(device)
        generated_ECG = G(amplitude, phase)[0, 0, :].cpu().detach().numpy()

        gen_ecgs.append(generated_ECG)

        time_x = np.linspace(0, 10, 2000)
        fig = plt.figure()
        plt.subplot(2, 1, 1)
        plt.plot(time_x, ecg, linewidth=1)
        plt.title('ECG')
        plt.subplot(2, 1, 2)
        plt.plot(time_x, generated_ECG, linewidth=1)
        plt.title('generated_ECG')
        fig.tight_layout()

        plt.savefig("images/temp/%s/%d.png" % (data_name, index), dpi=300)
        plt.close()

    np.save("images/temp/gen_ecgs.npy", np.asarray(gen_ecgs))


if __name__ == '__main__':
    model_path = 'weights/pix2pix_nature_99.pth'
    if opt.mode == 'train':
        train_main()
    elif opt.mode == 'test':
        test_main(model_path)
    elif opt.mode == 'eval_nature':
        data_path = '/home/wz/桌面/UWB2ECG/data/nature_data_resting_for_train/patient_data/GDN0001_1_Resting.mat'
        eval_nature(model_path, data_path)