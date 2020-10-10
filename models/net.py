from torch import nn


class FromRGB(nn.Module):
    def __init__(self, input_nc=3, ngf=64):
        super(FromRGB, self).__init__()

        # (h,w,input_nc=3) -> (h,w,nfg*4=256)
        blocks = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, kernel_size=7, stride=1, padding=0, bias=False),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(True),

            nn.ReflectionPad2d(1),
            nn.Conv2d(ngf, ngf * 2, kernel_size=3, stride=2, padding=0, bias=False),
            nn.InstanceNorm2d(ngf * 2),
            nn.ReLU(True),

            nn.ReflectionPad2d(1),
            nn.Conv2d(ngf * 2, ngf * 4, kernel_size=3, stride=2, padding=0, bias=False),
            nn.InstanceNorm2d(ngf * 4),
            nn.ReLU(True),
        ]

        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        x = self.blocks(x)
        return x
