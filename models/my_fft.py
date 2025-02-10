import torch
import torch.nn as nn
import torch.fft as fft

class ComplexFFT(nn.Module):
    def __init__(self):
        super(ComplexFFT, self).__init__()

    def forward(self, x):
        # 对输入进行复数FFT变换
        x_fft = fft.fft2(x, dim=(-2, -1))  # 2D FFT，沿最后两个维度进行
        real = x_fft.real  # 提取实部
        imag = x_fft.imag  # 提取虚部
        return real, imag


class ComplexIFFT(nn.Module):
    def __init__(self):
        super(ComplexIFFT, self).__init__()

    def forward(self, real, imag):
        # 复合实部和虚部，作为复杂信号
        x_complex = torch.complex(real, imag)
        # 进行复数IFFT逆变换
        x_ifft = fft.ifft2(x_complex, dim=(-2, -1))  # 2D IFFT
        return x_ifft.real  # 输出结果的实部


class Conv1x1(nn.Module):
    def __init__(self, in_channels):
        super(Conv1x1, self).__init__()
        self.conv = nn.Conv2d(in_channels * 2, in_channels * 2, kernel_size=1, stride=1, padding=0,groups=in_channels*2)

    def forward(self, x):
        return self.conv(x)


class Stage2_fft(nn.Module):
    def __init__(self, in_channels):
        super(Stage2_fft, self).__init__()
        self.c_fft = ComplexFFT()
        self.conv1x1 = Conv1x1(in_channels)
        self.c_ifft = ComplexIFFT()

    def forward(self, x):
        real, imag = self.c_fft(x)

        combined = torch.cat([real, imag], dim=1)
        conv_out = self.conv1x1(combined)

        out_channels = conv_out.shape[1] // 2
        real_out = conv_out[:, :out_channels, :, :]
        imag_out = conv_out[:, out_channels:, :, :]

        output = self.c_ifft(real_out, imag_out)

        return output

