import torch
from torch import nn
import math 
import random

time_steps = 1000
beta_schedule = torch.linspace(0.0001, 0.02, time_steps)

class conv3x3(nn.Module):
    def __init__(self, ic, oc):
        super().__init()
        self.conv = nn.Conv2d(ic, oc, 3)
    
    def forward(self, x):
        x = self.conv(x)
        x = nn.ReLU(x)
        return x 

class UNet():
    def __init__(self, block_type=conv3x3):
        super().__init()
        #self.positional_embedding = 

        self.d_1_1 = block_type(1, 64)
        self.d_1_2 = block_type(64, 64)
        self.mp_d_1 = nn.MaxPool2d(kernel_size = 2)

        self.d_2_1 = block_type(64, 128)
        self.d_2_2 = block_type(128, 128)
        self.mp_d_2 = nn.MaxPool2d(kernel_size = 2)

        self.d_3_1 = block_type(128, 256)
        self.d_3_2 = block_type(256, 256)
        self.mp_d_3 = nn.MaxPool2d(kernel_size = 2)

        self.d_4_1 = block_type(256, 512)
        self.d_4_2 = block_type(512, 512)
        self.mp_d_4 = nn.MaxPool2d(kernel_size = 2)

        self.m_1 = block_type(512, 1024)
        self.m_2 = block_type(1024, 1024)

        self.mp_u_4 = nn.ConvTranspose2d(1024, 1024, 4, 2, 1)
        self.u_4_1 = block_type(1024, 512)
        self.u_4_2 = block_type(512, 512)

        self.mp_u_3 = nn.ConvTranspose2d(512, 512, 4, 2, 1)
        self.u_3_1 = block_type(512, 256)
        self.u_3_2 = block_type(256, 256)

        self.mp_u_2 = nn.ConvTranspose2d(256, 256, 4, 2, 1)
        self.u_2_1 = block_type(256, 128)
        self.u_2_2 = block_type(128, 128)

        self.mp_u_1 = nn.ConvTranspose2d(128, 128, 4, 2, 1)
        self.u_1_1 = block_type(128, 64)
        self.u_1_2 = block_type(64, 64)

        self.out_layer = nn.Conv2d(64, 2, 1)

    def forward(self, x, t): 
        
        #x = self.positional_embedding(x)
        x = self.d_1_1(x)
        d1 = self.d_1_2(x) 
        x = self.mp_d_1(d1)

        x = self.d_2_1(x)
        d2 = self.d_2_2(x) 
        x = self.mp_d_2(d2) 

        x = self.d_3_1(x)
        d3 = self.d_3_2(x) 
        x = self.mp_d_2(d3)

        x = self.d_4_1(x)
        d4 = self.d_4_2(x) 
        x = self.mp_d_4(d4)

        x = self.m_1(x)
        x = self.m_2(x)

        x = self.mp_u_4(x) + d4
        x = self.u_4_1(x)
        x = self.u_4_2(x)

        x = self.mp_u_3(x) + d3
        x = self.u_3_1(x)
        x = self.u_3_2(x)

        x = self.mp_u_2(x) + d2
        x = self.u_2_1(x)
        x = self.u_2_2(x)
    
        x = self.mp_u_1(x) + d1
        x = self.u_1_1(x)
        x = self.u_1_2(x)

        x = self.out_layer(x)

        return x 

def alpha_dash_t(t): 
    return math.prod([1 -  beta_schedule[s] for s in range(t)])

def alpha_t(t):
    return 1 - beta_schedule[t]

def sample(e_model, noisy_image=torch.randn(1, 3, 720, 720)):
    for t in reversed(range(0, time_steps)):
        z = torch.rand_like(noisy_image) if t>0 else 0
        coeff = (1 - alpha_t(t))/ math.sqrt(1 - alpha_dash_t(t))
        noisy_image = (1/ math.sqrt(alpha_t(t))) * (noisy_image - coeff*e_model(noisy_image, t)) + beta_schedule[t]*z
    return noisy_image

def training(dataloader): 
    e_model = UNet()
    optimizer = torch.optim.Adam(e_model.parameters(), lr=1e-3)
    epochs = 2

    for epoch in range(epochs):
        for step, x_0 in enumerate(dataloader):            
            optimizer.zero_grad()

            t = random.randint(0, time_steps)
            e = torch.randn_like(x_0)
            e_theta = e_model(math.sqrt(alpha_dash_t(t))*x_0 + math.sqrt(1 - alpha_dash_t(t))*e, t)
            
            loss = nn.MSEloss(e, e_theta)
            loss.backward()
            optimizer.step()

