#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <myGridSampler.cuh>
#include <vector>
#include <stdio.h>

//using namespace std;
namespace{

template <typename scalar_t>
__global__ void bilinear_voxel_2d_kernel(
    const torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits, size_t> input,
    const torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits, size_t> grid,
    torch::PackedTensorAccessor<scalar_t, 4, torch::RestrictPtrTraits, size_t> output,
    torch::PackedTensorAccessor<float, 3, torch::RestrictPtrTraits, size_t> output_bilinear_count
)
{
    // input (N,C,H)
    // grid (N,H,Coor)
    // output (N,C, H, W)
    // output_bilinear_count (N,H,W)
    int C = input.size(1);
    int input_H = input.size(2);

    int out_H = output.size(2);
    int out_W = output.size(3);

    int grid_H=grid.size(1);
    int grid_Coor=grid.size(2);

        //batch index
    const int n = blockIdx.y;
    // column index
    const int h = blockIdx.x * blockDim.x + threadIdx.x;
    if(h < input_H){
        // get the corresponding input x, y co-ordinates from grid
      float ix = static_cast<float>(grid[n][h][0]);
      float iy = static_cast<float>(grid[n][h][1]);

        ix = grid_sampler_compute_source_index(ix, out_W);
        iy = grid_sampler_compute_source_index(iy, out_H);
        int ix0 = static_cast<int>(::floor(ix));
//        int ix1 = ::ceil(ix);
        int iy0 = static_cast<int>(::floor(iy));
//        int iy1 = ::ceil(iy);

//        float ix_rest = static_cast<float>(ix_nearest - ix);
//        float iy_rest = static_cast<float>(iy_nearest - iy);
        // hand-craft 4 points
        float weight =0;
        for (int i = ix0; i <= ix0+1; ++i){
            for (int j = iy0; j <= iy0+1; ++j){
                // assign nearest neighor pixel value to output pixel
                if (within_bounds_2d(j, i, out_H, out_W)) {
                    // bilinear: exp(-(ix**2+iy**2))

                    weight = (1.-::fabs(ix-i))*(1. - ::fabs(iy-j));
//# if __CUDA_ARCH__>=200
//    printf("weight, %f \n", weight);
//#endif
//                       float weight = 1.;
                //            atomicAdd((int* )&(output_count[n][iy_nearest][ix_nearest]), int(1));
                    atomicAdd((float* )&(output_bilinear_count[n][j][i]), 1.);
                //            safe_add_2d(count_ptr, iy_nearest, ix_nearest, out_ct_sH, out_ct_sW, out_H, out_W, 1);
                    for (int c = 0; c < C; ++c) {
                      // calculate and set grad_input
                      atomicAdd((scalar_t* )&(output[n][c][j][i]),weight*input[n][c][h]);
                    }
                }
            }
        }

    }
}

template <typename scalar_t>
__global__ void bilinear_voxel_2d_normal_kernel(
           torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits,size_t> output ,
           const torch::PackedTensorAccessor<float, 3, torch::RestrictPtrTraits, size_t> output_bilinear_count)
{
    // output (N,C, H, W)
    // output_count (N,H,W)
    int C = output.size(1);
    int out_H = output.size(2);
    int out_W = output.size(3);


        //batch index
      const int n = blockIdx.y;
      // column index
      const int hw = blockIdx.x * blockDim.x + threadIdx.x;
      const int h=hw/out_W;
      const int w=hw -h*out_W;
      if(h < out_H &&w < out_W){
        // get the corresponding input x, y co-ordinates from grid
        // assign nearest neighor pixel value to output pixel
        float bilinear_ct=output_bilinear_count[n][h][w];
        if(bilinear_ct>0){
            for (int c=0;c<C;c++){
                output[n][c][h][w]/=bilinear_ct;
            }
        }
      }
}

template <typename scalar_t>
__global__ void bilinear_voxel_2d_backward_kernel(
  const torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> grid,
  const torch::PackedTensorAccessor<float,3,torch::RestrictPtrTraits,size_t> output_bilinear_count,
  const torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits,size_t> grad_output,
  torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> grad_input
)
{

    // grid (N,H,Coor)
    // output_count (N, H, W)
    // grad_output (N,C,H,W)
    // grad_input (N,C,H2)

    int C = grad_output.size(1);
    int gInp_H = grad_input.size(2);

    int grid_H = grid.size(1);

    int out_H=output_bilinear_count.size(1);
    int out_W=output_bilinear_count.size(2);

        //batch index
      const int n = blockIdx.y;
      // column index
      const int h = blockIdx.x * blockDim.x + threadIdx.x;
      if(h < gInp_H){
            // get the corresponding input x, y co-ordinates from grid
          float ix = static_cast<float>(grid[n][h][0]);
          float iy = static_cast<float>(grid[n][h][1]);

          ix = grid_sampler_compute_source_index(ix, out_W);
          iy = grid_sampler_compute_source_index(iy, out_H);


        int ix0 = static_cast<int>(::floor(ix));
//        int ix1 = ::ceil(ix);
        int iy0 = static_cast<int>(::floor(iy));
        float weight =0.;
        // assign nearest neighor pixel value to output pixel
        for (int i = ix0; i <= ix0; ++i){
            for (int j = iy0; j <= iy0; ++j){
                auto ct= output_bilinear_count[n][j][i];
                if(ct<=0 || !within_bounds_2d(j, i, out_H, out_W)){
                    //TODO check here
                    for (int c = 0; c < C; ++c) {
                        grad_input[n][c][h] = static_cast<scalar_t>(0);
                    }
                }else{
                    for (int c = 0; c < C; ++c) {
                         weight  = (1.-::fabs(ix-i))*(1. - ::fabs(iy-j));
//                          float weight = 1.;
                //                    printf('%f',static_cast<float>(grad_output[n][c][iy_nearest][ix_nearest]/ct));
                        grad_input[n][c][h] = grad_output[n][c][j][i]*weight/ct;
                    }
                }
            }
        }
      }
}

}//namespace

// No shape checking needed here. See # NOTE [ grid_sampler Native Functions ].
std::tuple<torch::Tensor, torch::Tensor>
bilinear_grid_voxel_2d_cuda_forward(const torch::Tensor& input, const torch::Tensor& grid, torch::Tensor& output, torch::Tensor& output_bilinear_count) {
  const auto N = grid.size(0);
  const auto H = grid.size(1);

  const int threads=1024;
  const dim3 blocks((H+threads-1)/threads, N);

//    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "grid_voxel_2d_cuda", ([&] {
      bilinear_voxel_2d_kernel<float>
        <<<blocks,threads>>>(
          input.packed_accessor<float,3,torch::RestrictPtrTraits,size_t>(),
          grid.packed_accessor<float,3,torch::RestrictPtrTraits,size_t>(),
          output.packed_accessor<float,4,torch::RestrictPtrTraits,size_t>(),
          output_bilinear_count.packed_accessor<float,3,torch::RestrictPtrTraits,size_t>());
//    }));
         const auto out_H=output.size(2);
         const auto out_W=output.size(3);
        dim3 blocks2((out_H*out_W+threads-1)/threads, N);

       bilinear_voxel_2d_normal_kernel<float>
       <<<blocks2,threads>>>(
          output.packed_accessor<float,4,torch::RestrictPtrTraits,size_t>(),
          output_bilinear_count.packed_accessor<float,3,torch::RestrictPtrTraits,size_t>()
       );

  return std::make_tuple(output,output_bilinear_count);
};

// No shape checking needed here. See # NOTE [ grid_sampler Native Functions ].
torch::Tensor bilinear_grid_voxel_2d_cuda_backward(const torch::Tensor& grid, const torch::Tensor& output_bilinear_count,
                            const torch::Tensor& grad_output,torch::Tensor& grad_input) {
  const auto N = grid.size(0);
  const auto H = grid.size(1);

  const int threads=1024;
  const dim3 blocks((H+threads-1)/threads, N);


//    AT_DISPATCH_FLOATING_TYPES(output_bilinear_count.scalar_type(), "grid_voxel_2d_backward_cuda", ([&] {
      bilinear_voxel_2d_backward_kernel<float>
        <<<blocks,threads>>>(
              grid.packed_accessor<float,3,torch::RestrictPtrTraits,size_t>(),
              output_bilinear_count.packed_accessor<float,3,torch::RestrictPtrTraits,size_t>(),
              grad_output.packed_accessor<float,4,torch::RestrictPtrTraits,size_t>(),
              grad_input.packed_accessor<float,3,torch::RestrictPtrTraits,size_t>()
          );

//    }
//    ));
  return grad_input;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("bilinear_grid_voxel_2d_cuda_forward", &bilinear_grid_voxel_2d_cuda_forward, "bilinear_grid_voxel_2d_cuda_forward");
  m.def("bilinear_grid_voxel_2d_cuda_backward", &bilinear_grid_voxel_2d_cuda_backward, "bilinear_grid_voxel_2d_cuda_backward");
}