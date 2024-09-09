import torch
import torch.ao.quantization
import torch.nn as nn


class QuantizedConvReLU2d(nn.Module):
    r"""
    ReLu will be handled by output zero point properly.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        qw: torch.tensor,
        bias: torch.tensor,
        input_scale: float,
        input_zero: int,
        output_scale: float,
        output_zero: int,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        padding_mode="zeros",
        device=None,
        dtype=None,
    ):
        super(QuantizedConvReLU2d, self).__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=False,
            padding_mode=padding_mode,
            device=device,
            dtype=dtype,
        )

        self.qw = qw
        assert bias.numel() == 1 or bias.numel() == out_channels
        assert qw.shape[0] == out_channels and qw.shape[1] == in_channels
        assert qw.shape[2] == kernel_size[0] and qw.shape[3] == kernel_size[1]

        self.in_scale = input_scale
        self.in_zero = input_zero
        self.out_scale = output_scale
        self.out_zero = output_zero

        qw = self.qw.int_repr().to(torch.int32)
        if self.qw.qscheme() == torch.per_tensor_affine:
            qw_zero = self.qw.q_zero_point().to(torch.int32)
            self.w_scale = self.qw.q_scale()
        elif self.qw.qscheme() == torch.per_channel_affine:
            qw_zero = self.qw.q_per_channel_zero_points().to(torch.int32)
            self.w_scale = self.qw.q_per_channel_scales()

        with torch.no_grad():
            self.conv.weight = nn.Parameter(qw - qw_zero.view(-1, 1, 1, 1), requires_grad=False)

        scale_factor = self.in_scale * self.w_scale.view(1, -1, 1, 1) / self.out_scale
        assert torch.max(scale_factor) < 1.0 and torch.min(scale_factor) > 0.0
        self.scale_factor = torch.round(scale_factor * 2**31).to(torch.int32)

        q_bias = torch.round(bias / (self.in_scale * self.w_scale)).to(torch.int32)
        self.q_bias = q_bias.view(1, -1, 1, 1)

    def forward(self, input: torch.tensor) -> torch.tensor:
        if input.is_quantized:
            qi = input.int_repr()
        else:
            qi = torch.quantize_per_tensor(input, self.in_scale, self.in_zero, torch.quint8).int_repr()

        q_rel = (self.conv(qi.to(torch.int32) - self.in_zero) + self.q_bias).to(torch.int64)
        qo = ((self.scale_factor.to(torch.int64) * q_rel) >> 31) + self.out_zero

        qo = torch.clamp(qo, 0, 255)

        qr = (qo.to(torch.int32) - self.out_zero).to(torch.float32) * self.out_scale
        return torch.quantize_per_tensor(qr, self.out_scale, self.out_zero, torch.quint8)


class QuantizedLinearReLU(nn.Module):
    r"""
    ReLu will be handled by output zero point properly.
    """

    def __init__(
        self,
        in_features,
        out_features,
        qw: torch.tensor,
        bias: torch.tensor,
        input_scale: float,
        input_zero: int,
        output_scale: float,
        output_zero: int,
        dtype=None,
    ):
        super(QuantizedLinearReLU, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False, dtype=dtype)

        self.qw = qw
        assert bias.numel() == 1 or bias.numel() == out_features
        assert qw.shape[0] == out_features and qw.shape[1] == in_features

        self.in_scale = input_scale
        self.in_zero = input_zero
        self.out_scale = output_scale
        self.out_zero = output_zero

        qw = self.qw.int_repr().to(torch.int32)
        if self.qw.qscheme() == torch.per_tensor_affine:
            qw_zero = self.qw.q_zero_point().to(torch.int32)
            self.w_scale = self.qw.q_scale()
        elif self.qw.qscheme() == torch.per_channel_affine:
            qw_zero = self.qw.q_per_channel_zero_points().to(torch.int32)
            self.w_scale = self.qw.q_per_channel_scales()

        with torch.no_grad():
            self.linear.weight = nn.Parameter(qw - qw_zero.view(-1, 1), requires_grad=False)

        scale_factor = self.in_scale * self.w_scale.view(1, -1) / self.out_scale
        assert torch.max(scale_factor) < 1.0 and torch.min(scale_factor) > 0.0
        self.scale_factor = torch.round(scale_factor * 2**31).to(torch.int32)

        q_bias = torch.round(bias / (self.in_scale * self.w_scale)).to(torch.int32)
        self.q_bias = q_bias.view(1, -1)

    def forward(self, input: torch.tensor) -> torch.tensor:
        if input.is_quantized:
            qi = input.int_repr()
        else:
            qi = torch.quantize_per_tensor(input, self.in_scale, self.in_zero, torch.quint8).int_repr()
        qi = qi.contiguous().view(qi.shape[0], -1)
        
        q_rel = (self.linear(qi.to(torch.int32) - self.in_zero) + self.q_bias).to(torch.int64)
        qo = ((self.scale_factor.to(torch.int64) * q_rel) >> 31) + self.out_zero

        qo = torch.clamp(qo, 0, 255)
        qr = (qo.to(torch.int32) - self.out_zero).to(torch.float32) * self.out_scale
        return torch.quantize_per_tensor(qr, self.out_scale, self.out_zero, torch.quint8)
