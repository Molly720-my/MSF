# import torch
# import torch.nn as nn

# def feature_fusion(x, y):
#     """
#     对输入张量 x 进行压缩、分组和与 y 的特征融合。

#     参数:
#     x (torch.Tensor): 输入特征张量，形状为 [batch_size, 4096, 64]。
#     y (torch.Tensor): 参考特征张量，形状为 [batch_size, 256, 64]。

#     返回:
#     torch.Tensor: 经过压缩和融合后的特征张量，形状为 [batch_size, 256, 64]。
#     """
#     device = x.device  # Capture the device of input x to ensure consistency
#     y = y.to(device) 
#     # Define compression layers and preprocess layer based on the initial shape of x
#     initial_shape = x.shape[1]
#     if initial_shape == 4096:   
#         compression_layers = nn.Sequential(
#             nn.Conv1d(in_channels=4096, out_channels=2048, kernel_size=1),  # 压缩到 2048
#             nn.BatchNorm1d(2048), 
#             nn.ReLU(inplace=True), 
#             nn.Conv1d(in_channels=2048, out_channels=1024, kernel_size=1),  # 压缩到 1024
#             nn.BatchNorm1d(1024),
#             nn.ReLU(inplace=True),
#             nn.Conv1d(in_channels=1024, out_channels=512, kernel_size=1),  # 压缩到 512
#             nn.BatchNorm1d(512),
#             nn.ReLU(inplace=True),
#             nn.Conv1d(in_channels=512, out_channels=256, kernel_size=1),  # 最终压缩到 256
#             nn.BatchNorm1d(256),
#             nn.ReLU(inplace=True)
#         ).to(device) 

#         pre_conv = nn.Conv1d(in_channels=4096, out_channels=4096, kernel_size=1, groups=16)
#         x = pre_conv(x)  # 预处理后的形状 [batch_size, 4096, 64]

#     elif initial_shape == 1024:
#         compression_layers = nn.Sequential(
#             nn.Conv1d(in_channels=1024, out_channels=512, kernel_size=1),  # 压缩到 512
#             nn.BatchNorm1d(512),
#             nn.ReLU(inplace=True),
#             nn.Conv1d(in_channels=512, out_channels=256, kernel_size=1),  # 最终压缩到 256
#             nn.BatchNorm1d(256),
#             nn.ReLU(inplace=True)
#         ).to(device) 

#         pre_conv = nn.Conv1d(in_channels=1024, out_channels=1024, kernel_size=1, groups=16)
#         x = pre_conv(x)  # 预处理后的形状 [batch_size, 1024, 64]

#     elif initial_shape == 961:
#         compression_layers = nn.Sequential(
#             nn.Conv1d(in_channels=1024, out_channels=512, kernel_size=1),  # 压缩到 512
#             nn.BatchNorm1d(512),
#             nn.ReLU(inplace=True),
#             nn.Conv1d(in_channels=512, out_channels=256, kernel_size=1),  # 最终压缩到 256
#             nn.BatchNorm1d(256),
#             nn.ReLU(inplace=True)
#         ).to(device) 
#         # Set groups=1 because 961 is not divisible by 16
#         pre_conv = nn.Conv1d(in_channels=961, out_channels=1024, kernel_size=1, groups=1)
#         x = pre_conv(x)  # 预处理后的形状 [batch_size, 1024, 64]

#     # 分组，每组 [batch_size, 256, 64]
#     groups = torch.split(x, 256, dim=1)

#     # 定义多头注意力机制
#     attention = nn.MultiheadAttention(embed_dim=64, num_heads=8, batch_first=True).to(device) 

#     fused_groups = []

#     # 遍历每个组进行注意力融合
#     for group in groups:
#         fused_group, _ = attention(group.to("cuda"), y.to("cuda"), y.to("cuda"))
#         fused_groups.append(fused_group)

#     # 将融合后的组进行拼接
#     stacked = torch.cat(fused_groups, dim=1)  # 拼接后的形状 [batch_size, 4096, 64]
    
#     # 使用相应的压缩层对拼接后的特征进行压缩
#     x_compressed = compression_layers(stacked)  # 压缩后的形状 [batch_size, 256, 64]
#     print("use new!")
#     return x_compressed

# # # 示例输入张量
# # x = torch.randn(4, 961, 64).to("cuda")  # Move input to GPU
# # y = torch.randn(4, 256, 64).to("cuda")  # Move input to GPU

# # # 调用封装的函数
# # output = feature_fusion(x, y)
# # print(output.shape)  # 应输出 torch.Size([4, 256, 64])

import torch
import torch.nn as nn

def feature_fusion(x, y):
    """
    对输入张量 x 进行压缩、分组和与 y 的特征融合。
    
    参数:
    x (torch.Tensor): 输入特征张量，形状为 [batch_size, 4096, 64]。
    y (torch.Tensor): 参考特征张量，形状为 [batch_size, 256, 64]。

    返回:
    torch.Tensor: 经过压缩和融合后的特征张量，形状为 [batch_size, 256, 64]。
    """
    # Ensure x and y are on the same device
    device = x.device  
    y = y.to(device)  

    # Initialize the compression layer and preprocessing convolution based on input shape
    if x.shape[1] == 4096:   
        compression_layers = nn.Sequential(
            nn.Conv1d(in_channels=4096, out_channels=2048, kernel_size=1),
            nn.BatchNorm1d(2048), 
            nn.ReLU(inplace=True), 
            nn.Conv1d(in_channels=2048, out_channels=1024, kernel_size=1),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=1024, out_channels=512, kernel_size=1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=512, out_channels=256, kernel_size=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout1d(0.5)
        ).to(device)  

        pre_conv = nn.Conv1d(in_channels=4096, out_channels=4096, kernel_size=1, groups=16).to(device)
        x = pre_conv(x)

    elif x.shape[1] == 1024:
        compression_layers = nn.Sequential(
            nn.Conv1d(in_channels=1024, out_channels=512, kernel_size=1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=512, out_channels=256, kernel_size=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout1d(0.5)
        ).to(device)

        pre_conv = nn.Conv1d(in_channels=1024, out_channels=1024, kernel_size=1, groups=16).to(device)
        x = pre_conv(x)

    elif x.shape[1] == 961:
        compression_layers = nn.Sequential(
            nn.Conv1d(in_channels=1024, out_channels=512, kernel_size=1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=512, out_channels=256, kernel_size=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout1d(0.5)
        ).to(device)

        pre_conv = nn.Conv1d(in_channels=961, out_channels=1024, kernel_size=1, groups=1).to(device)
        x = pre_conv(x)

    elif x.shape[1] == 256:
        compression_layers = nn.Sequential(
            
            nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True)
        ).to(device)
    # Split into groups of [batch_size, 256, 64]
    groups = torch.split(x, 256, dim=1)

    # Define multi-head attention mechanism
    attention = nn.MultiheadAttention(embed_dim=64, num_heads=8, batch_first=True).to(device)

    fused_groups = []

    # Apply attention fusion for each group
    for group in groups:
        fused_group, _ = attention(group, y, y)
        fused_groups.append(fused_group)

    # Concatenate the fused groups
    stacked = torch.cat(fused_groups, dim=1)  # Shape: [batch_size, 4096, 64] (if original was 4096)

    # Apply the appropriate compression layers
    # print("stacked",stacked.shape)
    x_compressed = compression_layers(stacked)

    return x_compressed

# # Example input tensors
# x = torch.randn(8, 961, 64).to("cuda")  # Move input to GPU
# y = torch.randn(8, 256, 64).to("cuda")  # Move input to GPU

# # Call the function
# output = feature_fusion(x, y)
# print(output.shape)  # Expected output shape: torch.Size([4, 256, 64])
