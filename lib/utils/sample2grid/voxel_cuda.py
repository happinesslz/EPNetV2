"""
Created by silver at 2019/10/15 22:29
Email: xiwuchencn[at]gmail[dot]com
"""

import torch
from gridvoxel_cuda import grid_voxel_2d_cuda_forward, grid_voxel_2d_cuda_backward
from gaussian_gridvoxel_cuda import gaussian_grid_voxel_2d_cuda_forward,gaussian_grid_voxel_2d_cuda_backward
from bilinear_gridvoxel_cuda import bilinear_grid_voxel_2d_cuda_forward,bilinear_grid_voxel_2d_cuda_backward

from torch.autograd import Function



class Sample2Grid(Function):
    @staticmethod
    def forward(ctx, input, grid, output_size):
        """
        This function is the inverse operation of grid_sample. While grid_sample sampling value from img(4D) or
        some(5D) tensor by grid(dims same as the input).
        But this function is specify modified for the point cloud. the grid and input's dim(3D) is less than output(4D).
        :param ctx:
        :param input: (B,C,N)
        :param grid: (B,N,Coor)
        :param output_size: [b,c,h,w]
        :return: output: (B,C,H,W); output_ct: (B,H,W)
        """
        # assert grid.requires_grad == False
        assert input.dim()==3
        assert grid.dim()==3
        assert output_size.__len__()==4
        assert output_size[1]==input.size(1)
        # print(input.requires_grad)
        output = torch.zeros(output_size, device = input.device).float().requires_grad_(False)
        output_ct = torch.zeros([output_size[i] for i in [0, 2, 3]], device = input.device, dtype = torch.int)
        grid_voxel_2d_cuda_forward(input, grid, output, output_ct)
        ctx.save_for_backward(grid, output_ct)
        return output.requires_grad_(True)

    @staticmethod
    def backward(ctx, grad_output):
        # print('grad_output',grad_output)
        C = grad_output.size(1)
        grid, output_ct = ctx.saved_tensors
        B, N = grid.shape[0:2]
        d_input = torch.zeros((B, C, N), device = grid.device,dtype = torch.float)
        grid_voxel_2d_cuda_backward(grid, output_ct, grad_output, d_input)

        return d_input, None, None

sample2grid_F = Sample2Grid.apply

class Sample2GaussianGrid(Function):
    @staticmethod
    def forward(ctx, input, grid, output_size):
        """
        This function is the inverse operation of grid_sample. While grid_sample sampling value from img(4D) or
        some(5D) tensor by grid(dims same as the input).
        But this function is specify modified for the point cloud. the grid and input's dim(3D) is less than output(4D).
        :param ctx:
        :param input: (B,C,N)
        :param grid: (B,N,Coor)
        :param output_size: [b,c,h,w]
        :return: output: (B,C,H,W); output_ct: (B,H,W)
        """
        # assert grid.requires_grad == False
        assert input.dim()==3
        assert grid.dim()==3
        assert output_size.__len__()==4
        assert output_size[1]==input.size(1)
        # print(input.requires_grad)
        output = torch.zeros(output_size, device = input.device).float().requires_grad_(False)
        output_gaussian_ct = torch.zeros([output_size[i] for i in [0, 2, 3]], device = input.device,dtype = torch.float)
        gaussian_grid_voxel_2d_cuda_forward(input, grid, output, output_gaussian_ct)
        ctx.save_for_backward(grid, output_gaussian_ct)
        return output.requires_grad_(True)

    @staticmethod
    def backward(ctx, grad_output):
        # print('grad_output',grad_output)
        C = grad_output.size(1)
        grid, output_gaussian_ct = ctx.saved_tensors
        B, N = grid.shape[0:2]
        d_input = torch.zeros((B, C, N), device = grid.device,dtype = torch.float)
        gaussian_grid_voxel_2d_cuda_backward(grid, output_gaussian_ct, grad_output, d_input)

        return d_input, None, None

sample2GaussianGrid_F = Sample2GaussianGrid.apply

class Sample2BilinearGrid(Function):
    @staticmethod
    def forward(ctx, input, grid, output_size):
        """
        This function is the inverse operation of grid_sample. While grid_sample sampling value from img(4D) or
        some(5D) tensor by grid(dims same as the input).
        But this function is specify modified for the point cloud. the grid and input's dim(3D) is less than output(4D).
        :param ctx:
        :param input: (B,C,N)
        :param grid: (B,N,Coor)
        :param output_size: [b,c,h,w]
        :return: output: (B,C,H,W); output_ct: (B,H,W)
        """
        # assert grid.requires_grad == False
        assert input.dim()==3
        assert grid.dim()==3
        assert output_size.__len__()==4
        assert output_size[1]==input.size(1)
        # print(input.requires_grad)
        output = torch.zeros(output_size, device = input.device).float().requires_grad_(False)
        output_gaussian_ct = torch.zeros([output_size[i] for i in [0, 2, 3]], device = input.device,dtype = torch.float)
        bilinear_grid_voxel_2d_cuda_forward(input, grid, output, output_gaussian_ct)
        ctx.save_for_backward(grid, output_gaussian_ct)
        return output.requires_grad_(True)

    @staticmethod
    def backward(ctx, grad_output):
        # print('grad_output',grad_output)
        C = grad_output.size(1)
        grid, output_gaussian_ct = ctx.saved_tensors
        B, N = grid.shape[0:2]
        d_input = torch.zeros((B, C, N), device = grid.device,dtype = torch.float)
        bilinear_grid_voxel_2d_cuda_backward(grid, output_gaussian_ct, grad_output, d_input)

        return d_input, None, None

sample2BilinearGrid_F = Sample2BilinearGrid.apply


if __name__ == '__main__':
    from torch.nn.functional import grid_sample
    import torch
    B = 1
    C = 1
    H, W = 2,2
    N = 1
    # img = torch.rand([1, 1, 5, 5]).cuda().requires_grad_(True)
    # index=torch.randint(0,5,size = [1,C,2])
    # grid=index.float()/(torch.tensor([5.,5.])-1.)*2-1.
    seed = 0
    torch.manual_seed(seed)  # 为CPU设置随机种子
    torch.cuda.manual_seed(seed)  # 为当前GPU设置随机种子
    torch.cuda.manual_seed_all(seed)
    grid = torch.rand([B, N, 2]) * 2 - 1.
    # print(grid)
    # print(index)
    # index_ct=torch.zeros([1,5,5])
    # for i in range(C):
    #     index_ct[0,index[0,i,1],index[0,i,0]]+=1
    # print('index',index_ct)
    grid = grid.cuda().requires_grad_(True)
    # pc = grid_sample(img, grid.unsqueeze(-2),mode = 'nearest',padding_mode ='zeros')
    # print(pc.shape)
    # pc=pc.squeeze(-1)
    pc=torch.rand([B, C, N]).cuda().requires_grad_(True)
    # print(pc.requires_grad)
    # pc=pc.requires_grad_(True)
    img_new = sample2grid_F(pc, grid.clone(), [B, C, H, W])
    print(grid)
    print("1111111111111111")
    print(img_new)
    print(pc)
    loss=(img_new**2).sum()
    loss.backward()
    print(img_new.grad)
    print('pc_grad',pc.grad)

    print("2222222222222222")
    img_new2=sample2GaussianGrid_F(pc,grid.clone(),[B, C, H, W])

    print(img_new2)
    print(pc)
    pc.grad.zero_()
    loss=(img_new2**2).sum()
    loss.backward()
    print(img_new2.grad)
    print('pc_grad',pc.grad)

    print("3333333333333333")
    img_new3 = sample2BilinearGrid_F(pc, grid.clone(), [B, C, H, W])
    print(img_new3)
    print(pc)
    pc.grad.zero_()
    loss=(img_new3**2).sum()
    loss.backward()
    print(img_new3.grad)
    print('pc_grad',pc.grad)



