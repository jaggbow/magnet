from torch import nn

class ResBlock(nn.Module):
    def __init__(self, n_chan, kernel_size,
        bias=True, act=nn.ReLU(True), res_scale=1, mode='1d'):
        
        super().__init__()

        assert mode in ["1d", "2d"]
        self.res_scale = res_scale
        self.mode = mode
        
        if mode == '2d':
            self.conv_1 = nn.Conv2d(n_chan, n_chan, kernel_size, padding=(kernel_size//2))
            self.act = act
            self.conv_2 = nn.Conv2d(n_chan, n_chan, kernel_size, padding=(kernel_size//2))
        else:
            self.conv_1 = nn.Conv1d(n_chan, n_chan, kernel_size, padding=(kernel_size//2))
            self.act = act
            self.conv_2 = nn.Conv1d(n_chan, n_chan, kernel_size, padding=(kernel_size//2))
    
    def forward(self, x):
        out = self.conv_1(x)
        out = self.act(out)
        out = self.conv_2(out)
        
        out += x
        
        out = out.mul(self.res_scale)
        return out

class EDSR(nn.Module):
    def __init__(self, in_chan, n_chan=64, res_layers=16, kernel_size=3, res_scale=1, mode='1d'):
        '''
        EDSR model without upsampling
        '''
        super().__init__()
        assert mode in ["1d", "2d"]
        self.mode = mode

        if mode == '2d':
            self.head_conv = nn.Conv2d(in_chan, n_chan, kernel_size, padding=(kernel_size//2))
            self.res_layers = nn.Sequential(*[ResBlock(n_chan, kernel_size, res_scale, mode=mode) for _ in range(res_layers)])
            self.tail_conv = nn.Conv2d(n_chan, n_chan, kernel_size, padding=(kernel_size//2))
        else:
            self.head_conv = nn.Conv1d(in_chan, n_chan, kernel_size, padding=(kernel_size//2))
            self.res_layers = nn.Sequential(*[ResBlock(n_chan, kernel_size, res_scale, mode=mode) for _ in range(res_layers)])
            self.tail_conv = nn.Conv1d(n_chan, n_chan, kernel_size, padding=(kernel_size//2))
        
        self.out_dim = n_chan
    
    def forward(self, x):


        x = self.head_conv(x)
        res = self.res_layers(x)
        res = self.tail_conv(res)
        res += x

        return res