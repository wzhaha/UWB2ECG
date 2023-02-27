"""
add contrasive learning module
"""
from torch.utils.data import DataLoader
import argparse
from utils.datautil import *
import torch.nn as nn
from utils.evaluation_matrix import *
from torch.optim.lr_scheduler import StepLR


parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default='pix2pix_nature_contra', help="name of model")
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
parser.add_argument("--mode", type=str, default='test', help="test train eval_uwb")

opt = parser.parse_args()
print(opt)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
# ecgs, win_amplitude_datas, win_phase_datas = load_data(opt)
# _GDN0001
app = '_GDN0006'
train_dataset_file = 'data/nature_data_resting_for_train/train_nature%s.npy' % (app)
test_dataset_file = 'data/nature_data_resting_for_train/test_nature%s.npy' % (app)

if opt.mode == 'train':
    # train_data, test_data = trainTestSplit(ecgs, win_amplitude_datas, win_phase_datas, opt.train_rate)
    train_data = np.load(train_dataset_file)
    test_data = np.load(test_dataset_file)
    train_dataset = UWBECG_Contra_DataSet(train_data)
    train_dataloader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True)
    test_dataset = UWBECG_Contra_DataSet(test_data)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)
elif opt.mode == 'test':
    # train_data, test_data = trainTestSplit(ecgs, win_amplitude_datas, win_phase_datas, 0)
    test_data = np.load(test_dataset_file)
    test_dataset = UWBECG_Contra_DataSet(test_data)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)


def sample_signals(epoch, batch):
    os.makedirs('images/' + opt.model_name + '/train', exist_ok=True)

    data = next(iter(test_dataloader))
    G.eval()

    ecg = data[0].type(torch.FloatTensor).to(device)
    amplitude_data = data[1].type(torch.FloatTensor).to(device)
    phase_data = data[2].type(torch.FloatTensor).to(device)
    sine_amp = data[3].type(torch.FloatTensor).to(device)
    sine_pha = data[4].type(torch.FloatTensor).to(device)

    _, generated_ECG, _, _ = G(amplitude_data, phase_data, sine_amp, sine_pha)

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
            nn.Conv1d(1, 8, kernel_size=31, padding=pad31, ),
            nn.InstanceNorm1d(opt.sample_len),
            nn.LeakyReLU(opt.leaky_relu_param,),
            # nn.Dropout(0.3),
            nn.Conv1d(8, 16, kernel_size=31, padding=pad31, ),
            nn.InstanceNorm1d(opt.sample_len),
            nn.LeakyReLU(opt.leaky_relu_param,),
            # nn.Dropout(0.3),
            nn.Conv1d(16, 32, kernel_size=31, padding=pad31, ),
            nn.InstanceNorm1d(opt.sample_len),
            nn.LeakyReLU(opt.leaky_relu_param,),
            # nn.Dropout(0.3),
            nn.Conv1d(32, 64, kernel_size=31, padding=pad31, ),
            nn.InstanceNorm1d(opt.sample_len),
            nn.LeakyReLU(opt.leaky_relu_param, ),
            # nn.Dropout(0.3),
        )

        self.encoder2 = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=31, padding=pad31, ),
            nn.InstanceNorm1d(opt.sample_len),
            nn.LeakyReLU(opt.leaky_relu_param,),
            # nn.Dropout(0.3),
            nn.Conv1d(8, 16, kernel_size=31, padding=pad31, ),
            nn.InstanceNorm1d(opt.sample_len),
            nn.LeakyReLU(opt.leaky_relu_param,),
            # nn.Dropout(0.3),
            nn.Conv1d(16, 32, kernel_size=31, padding=pad31, ),
            nn.InstanceNorm1d(opt.sample_len),
            nn.LeakyReLU(opt.leaky_relu_param,),
            # nn.Dropout(0.3),
            nn.Conv1d(32, 64, kernel_size=31, padding=pad31, ),
            nn.InstanceNorm1d(opt.sample_len),
            nn.LeakyReLU(opt.leaky_relu_param, ),
            # nn.Dropout(0.3),
        )

        self.decoder = nn.Sequential(
            nn.Conv1d(128, 64, kernel_size=5, padding=pad5, ),
            nn.InstanceNorm1d(opt.sample_len),
            nn.LeakyReLU(opt.leaky_relu_param, ),
            # nn.Dropout(0.3),
            nn.Conv1d(64, 32, kernel_size=9, padding=pad9, ),
            nn.InstanceNorm1d(opt.sample_len),
            nn.LeakyReLU(opt.leaky_relu_param, ),
            # nn.Dropout(0.3),
            nn.Conv1d(32, 16, kernel_size=9, padding=max(9 - 1, 0) // 2, ),
            nn.InstanceNorm1d(opt.sample_len),
            nn.LeakyReLU(opt.leaky_relu_param, ),
            # nn.Dropout(0.3),
            nn.Conv1d(16, 8, kernel_size=9, padding=max(9 - 1, 0) // 2, ),
            nn.InstanceNorm1d(opt.sample_len),
            nn.LeakyReLU(opt.leaky_relu_param, ),
            # nn.Dropout(0.3),
            nn.Conv1d(8, 4, kernel_size=21, padding=max(21 - 1, 0) // 2, ),
            nn.InstanceNorm1d(opt.sample_len),
            nn.LeakyReLU(opt.leaky_relu_param, ),
            # nn.Dropout(0.3),
            nn.Conv1d(4, 1, kernel_size=31, padding=max(31 - 1, 0) // 2, ),
            nn.Tanh(),
        )

    def forward(self, amplitude, angle, sine_amp, sine_pha):
        x1 = self.encoder1(amplitude)
        x2 = self.encoder1(angle)
        x = torch.cat((x1, x2), 1)
        out = self.decoder(x)

        x1_sine = self.encoder1(sine_amp)
        x2_sine = self.encoder1(sine_pha)
        x_sine = torch.cat((x1_sine, x2_sine), 1)
        out_sine = self.decoder(x_sine)
        return x, out, x_sine, out_sine


G = Pix2Pix(opt).to(device)
D = Discriminator(opt).to(device)


def train_main():
    print("start to train")

    optimizerG = torch.optim.Adam(G.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizerD = torch.optim.Adam(D.parameters(), lr=0.0001, betas=(opt.b1, opt.b2))

    schedulerG = StepLR(optimizer=optimizerG, step_size=5, gamma=0.8)
    schedulerD = StepLR(optimizer=optimizerD, step_size=5, gamma=0.8)

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
            sine_amp = data[3].type(torch.FloatTensor).to(device)
            sine_pha = data[4].type(torch.FloatTensor).to(device)

            real_label = torch.ones(ecg.shape[0]).to(device)  # 定义真实的图片label为1
            fake_label = torch.zeros(ecg.shape[0]).to(device)  # 定义假的图片的label为0

            # (1) Update G network: Generators
            for k in range(1):
                optimizerG.zero_grad()
                # GAN loss
                h1, fake_ECG, h2, _ = G(amplitude_data, phase_data, sine_amp, sine_pha)
                pred_fake = D(fake_ECG, amplitude_data, phase_data)
                loss_GAN = MSECriterion(pred_fake, real_label)
                # Pixel-wise loss
                loss_pixel = L1Loss(fake_ECG, ecg)
                # # contrasive loss
                # contra_loss = MSECriterion(h1, h2) * 0.05
                contra_loss=0
                loss_G = loss_GAN + opt.lambda_pixel * loss_pixel + contra_loss
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
            # nn.utils.clip_grad_norm(D.parameters(), max_norm=10, norm_type=2)
            optimizerD.step()

            d_accuracy = sum(train_labels.cpu().detach().numpy() == np.int64(np.asarray(pred_D.cpu().detach().numpy()) > 0.5))/train_labels.shape[0]

            if (i + 1) % 5 == 0:
                print('E[%d/%d],B[%d/%d], [Dloss:%.2f Dacc:%2f][Gloss:%.2f, adv:%.2f, pix:%.2f, contra_loss:%2f,Gacc:%2f]' % (
                    epoch+1, opt.epochs, i, len(train_dataloader),
                    loss_D.item(),
                    d_accuracy,
                    loss_G.item(),
                    loss_GAN.item(),
                    loss_pixel.item(),
                    contra_loss,
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
        if epoch % 2 == 1:
            torch.save(G.state_dict(), "weights/%s_%d.pth"%(opt.model_name, epoch))

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
        ecgs.append(data[0].detach().numpy()[0, 0, :])
        amplitude_data = data[1].type(torch.FloatTensor).to(device)
        phase_data = data[2].type(torch.FloatTensor).to(device)
        sine_amp = data[3].type(torch.FloatTensor).to(device)
        sine_pha = data[4].type(torch.FloatTensor).to(device)

        _, generated_ECG, _, _ = G(amplitude_data, phase_data, sine_amp, sine_pha)
        generated_ECG = generated_ECG[0, 0, :].cpu().detach().numpy()
        gen_ecgs.append(generated_ECG)

        gt_ecg = ecg[0, 0, :].cpu().numpy()

        matlab_xcorr(generated_ECG, gt_ecg)

        cc_score = cc(generated_ECG, gt_ecg)
        cc_scores.append(cc_score)
        cos_dis_score = cos_dis(generated_ECG, gt_ecg)
        cos_scores.append(cos_dis_score)
        rmse_score = rmse(generated_ECG, gt_ecg)
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
        #
        # plt.savefig("images/%s/test/pic/%d.png" % (opt.model_name, index))
        # plt.close()

    np.save("images/%s/test/npy/ecgs.npy"% (opt.model_name), np.asarray(ecgs))
    np.save("images/%s/test/npy/gen_ecgs.npy" % (opt.model_name), np.asarray(gen_ecgs))
    # print('end to test')

    # cc_scores = np.asarray(cc_scores)
    # print('median cc score', np.mean(cc_scores))

    cos_scores = np.asarray(cos_scores)
    print('median cos score', np.median(cos_scores))

    rmse_scores = np.asarray(rmse_scores)
    print('median rmse score', np.median(rmse_scores))

    np.save('images/temp/cos_scores_nature%s.npy' % (app), cos_scores)
    np.save('images/temp/rmse_scores_nature%s.npy' % (app), rmse_scores)
    # np.save('images/temp/cc_scores_nature%s.npy' % (app), cc_scores)

    median_err = evaluate_ecg_interval('images/%s/test/npy/' % (opt.model_name))
    return median_err


if __name__ == '__main__':
    if opt.mode == 'train':
        train_main()
    elif opt.mode == 'test':
        model_path = 'weights/fold_eval_nature/pix2pix_nature_6_99.pth'
        median_err = test_main(model_path)

        # min_index = -1
        # min_err = 100
        # for model_index in range(93, 73, -2):
        #     # model_index = 97
        #     model_path = 'weights/pix2pix_nature_contra_%d.pth' % (model_index)
        #     median_err = test_main(model_path)
        #     if sum(median_err) < min_err:
        #         min_err = sum(median_err)
        #         min_index = model_index
        #
        # print('best index:', min_index)