import torch.nn.functional as F

def flatten(x):
    N = x.shape[0] # read in N, C, H, W
    return x.view(N, -1)  # "flatten" the C * H * W values into a single vector per image

def two_layer_fc(x, params):
    x = flatten(x)  # shape: [batch_size, C x H x W]
    w1, w2 = params
    x = F.relu(x.mm(w1))
    x = x.mm(w2)
    return x

def three_layer_convnet(x, params):
    conv_w1, conv_b1, conv_w2, conv_b2, fc_w, fc_b = params
    scores = None
    z1 = F.conv2d(x, conv_w1, conv_b1, padding='same')
    a1 = F.relu(z1)
    z2 = F.conv2d(a1, conv_w2, conv_b2, padding='same')
    a2 = F.relu(z2)
    fc = flatten(a2)
    scores = fc.mm(fc_w) + fc_b
    return scores