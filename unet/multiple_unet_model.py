""" Full assembly of the parts to form the complete network """

from .unet_parts import *


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 4 if bilinear else 3
        self.down4 = (Down(512, 1024 // factor))
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, n_classes))

    def forward(self, x):
        x = [x[0], x[1], x[2], x[3]]
        x1 = []
        x2 = []
        x3 = []
        x4 = []
        x5 = []
        
        for i in range(len(x)):
            x1.append(self.inc(x[i]))
            x2.append(self.down1(x1[i]))
            x3.append(self.down2(x2[i]))
            x4.append(self.down3(x3[i]))
            x5.append(self.down4(x4[i]))
            
        x1 = torch.cat(x1, dim=1)
        x2 = torch.cat(x2, dim=1)
        x3 = torch.cat(x3, dim=1)
        x4 = torch.cat(x4, dim=1)
        
        x_out = []
        for i in range(len(x5)):     
            x_out.append(self.up1(x5[i], x4))
            x_out[i] = self.up2(x_out[i], x3)
            x_out[i] = self.up3(x_out[i], x2)
            x_out[i] = self.up4(x_out[i], x1)
            x_out[i] = self.outc(x_out[i])
        return x_out

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)