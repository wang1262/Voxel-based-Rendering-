
#include "voxelization.h"
#include <glm/glm.hpp>
#include <GL/glut.h>

#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>

#include <voxelpipe/voxelpipe.h>

voxelpipe::FRContext<log_N, log_T>*  context;
bool first_time = true;

struct ColorShader
{
  int* texture;
  int tex_width;
  int tex_height;
  float* texcoord;
  int texcoord_size;

  __device__ float shade(
    const int tri_id,
    const float4 v0,
    const float4 v1,
    const float4 v2,
    const float3 n,
    const float  bary0,
    const float  bary1,
    const int3   xyz) const
  {
    //Return G = 255, A = 255 for now
    return __int_as_float((255 << 7) + (255 << 23));

    //TODO: Pull the color from the texture using the barycentric/texcoords
  }
};

template<typename storage_type>
__global__ void getOccupiedVoxels(void* fb, int M, int T, int* voxels) {
  int T3 = T*T*T;
  int M3 = M*M*M;

  int pix_num = (blockIdx.x * 256 % T3) + threadIdx.x;
  int tile_num = blockIdx.x * 256 / T3;

  if (pix_num < T3 && tile_num < M3) {
    //TODO: Is there any benefit in making this shared?
    storage_type* tile;

    bool is_occupied;
    if (T <= 8 && BIT_MODE) {
      tile = (storage_type*)fb + tile_num*T3;
      is_occupied = tile[pix_num];
    } else if (BIT_MODE) {
      tile = (storage_type*)fb + tile_num*(T3 >> 5);
      is_occupied = (tile[pix_num >> 5] & (1 << (pix_num & 31)));
    } else {
      tile = (storage_type*)fb + tile_num*T3;
      int alpha = ((int)tile[pix_num]) >> 23;
      is_occupied = alpha > 0;
    }

    if (is_occupied) {
      voxels[tile_num*T3 + pix_num] = tile_num*T3 + pix_num;
    } else {
      voxels[tile_num*T3 + pix_num] = -1;
    }
  }

}

//Thrust predicate for removal of empty voxels
struct check_voxel {
  __host__ __device__
    bool operator() (const int& c) {
    return (c != -1);
  }
};

__global__ void extractValues(void* fb, int* voxels, int num_voxels, int* values) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;

  if (index < num_voxels) {
    //TODO: Make this support other storage_type's besides int32
    float* tile = (float*)fb;
    values[index] = __float_as_int(tile[voxels[index]]);
  }
}

__global__ void createCubeMesh(int* voxels, int* values, int M, int T, float3 bbox0, float3 t_d, float3 p_d, float scale_factor, int num_voxels, float* cube_vbo, 
                                int cube_vbosize, int* cube_ibo, int cube_ibosize, float* cube_nbo, float* out_vbo, int* out_ibo, float* out_nbo, float* out_cbo) {

  //Get the index for the thread
  int idx = blockIdx.x*blockDim.x + threadIdx.x;

  if (idx < num_voxels) {

    int vbo_offset = idx * cube_vbosize;
    int ibo_offset = idx * cube_ibosize;
    float3 center = getCenterFromIndex(voxels[idx], M, T, bbox0, t_d, p_d);
    int color = values[idx];

    for (int i = 0; i < cube_vbosize; i++) {
      if (i % 3 == 0) {
        out_vbo[vbo_offset + i] = cube_vbo[i] * scale_factor + center.x;
        out_cbo[vbo_offset + i] = (float)((color & 0xFF)/255.0);
      } else if (i % 3 == 1) {
        out_vbo[vbo_offset + i] = cube_vbo[i] * scale_factor + center.y;
        out_cbo[vbo_offset + i] = (float)(((color >> 7) & 0xFF)/255.0);
      } else {
        out_vbo[vbo_offset + i] = cube_vbo[i] * scale_factor + center.z;
        out_cbo[vbo_offset + i] = (float)(((color >> 15) & 0xFF)/255.0);
      }
      out_nbo[vbo_offset + i] = cube_nbo[i];
    }

    for (int i = 0; i < cube_ibosize; i++) {
      out_ibo[ibo_offset + i] = cube_ibo[i] + ibo_offset;
    }

  }

}

__host__ int voxelizeMesh(Mesh &m_in, int* d_voxels, int* d_values) {

  //Initialize sizes
  const int n_triangles = m_in.ibosize / 3;
  const int n_vertices = m_in.vbosize / 3;

  //Create host vectors
  thrust::host_vector<int4> h_triangles(n_triangles);
  thrust::host_vector<float4> h_vertices(n_vertices);

  //Fill in the data
  for (int i = 0; i < n_vertices; i++) {
    h_vertices[i].x = m_in.vbo[i * 3 + 0];
    h_vertices[i].y = m_in.vbo[i * 3 + 1];
    h_vertices[i].z = m_in.vbo[i * 3 + 2];
  }
  for (int i = 0; i < n_triangles; i++) {
    h_triangles[i].x = m_in.ibo[i * 3 + 0];
    h_triangles[i].y = m_in.ibo[i * 3 + 1];
    h_triangles[i].z = m_in.ibo[i * 3 + 2];
  }

  //Copy to device vectors
  thrust::device_vector<int4> d_triangles(h_triangles);
  thrust::device_vector<float4> d_vertices(h_vertices);

  if (first_time) {
    //Create the voxelpipe context
    context = new voxelpipe::FRContext<log_N, log_T>();

    //Reserve data for voxelpipe
    context->reserve(n_triangles, 1024u * 1024u * 16u);
  }
  first_time = false;

  //Define types for the voxelization
  typedef voxelpipe::FR::TileOp<voxelpipe::Bit, voxelpipe::ADD_BLENDING, log_T> tile_op_type;
  typedef typename tile_op_type::storage_type storage_type;

  //Initialize the result data on the device
  thrust::device_vector<storage_type>  d_fb(M*M*M * tile_op_type::STORAGE_SIZE);

  //Create the shader to be used that will write texture colors to voxels
  ColorShader my_shader;
  my_shader.texture = NULL;
  my_shader.tex_height = 0;
  my_shader.tex_width = 0;
  my_shader.texcoord = m_in.tbo;
  my_shader.texcoord_size = m_in.tbosize;

  //Perform coarse and fine voxelization
  context->coarse_raster(n_triangles, n_vertices, thrust::raw_pointer_cast(&d_triangles.front()), thrust::raw_pointer_cast(&d_vertices.front()), bbox0, bbox1);
  if (BIT_MODE) {
    context->fine_raster< voxelpipe::Bit, voxelpipe::BIT_FORMAT, voxelpipe::THIN_RASTER, voxelpipe::ADD_BLENDING, voxelpipe::DefaultShader<voxelpipe::Bit> >(
      n_triangles, n_vertices, thrust::raw_pointer_cast(&d_triangles.front()), thrust::raw_pointer_cast(&d_vertices.front()), bbox0, bbox1, thrust::raw_pointer_cast(&d_fb.front()));
  } else {
    context->fine_raster< voxelpipe::Float, voxelpipe::FP32S_FORMAT, voxelpipe::THIN_RASTER, voxelpipe::MAX_BLENDING, ColorShader >(
      n_triangles, n_vertices, thrust::raw_pointer_cast(&d_triangles.front()), thrust::raw_pointer_cast(&d_vertices.front()), bbox0, bbox1, thrust::raw_pointer_cast(&d_fb.front()), my_shader);
  }

  //Get voxel centers
  int numVoxels = N*N*N;
  int* d_vox;
  cudaMalloc((void**)&d_vox, numVoxels*sizeof(int));
  getOccupiedVoxels<storage_type> << <N*N*N, 256 >> >(thrust::raw_pointer_cast(&d_fb.front()), M, T, d_vox);
  cudaDeviceSynchronize();

  //Stream Compact voxels to remove the empties
  numVoxels = thrust::copy_if(thrust::device_pointer_cast(d_vox), thrust::device_pointer_cast(d_vox) + numVoxels, thrust::device_pointer_cast(d_voxels), check_voxel()) - thrust::device_pointer_cast(d_voxels);

  //Extract the values at these indices
  extractValues<<<(numVoxels / 256) + 1, 256 >>>(thrust::raw_pointer_cast(&d_fb.front()), d_voxels, numVoxels, d_values);
  cudaDeviceSynchronize();

  cudaFree(d_vox);

  return numVoxels;
}

__host__ void extractCubesFromVoxelGrid(int* d_voxels, int numVoxels, int* d_values, Mesh &m_cube, Mesh &m_out) {

  //Move cube data to GPU
  thrust::device_vector<float> d_vbo_cube(m_cube.vbo, m_cube.vbo + m_cube.vbosize);
  thrust::device_vector<int> d_ibo_cube(m_cube.ibo, m_cube.ibo + m_cube.ibosize);
  thrust::device_vector<float> d_nbo_cube(m_cube.nbo, m_cube.nbo + m_cube.nbosize);

  //Create output structs
  float* d_vbo_out;
  int* d_ibo_out;
  float* d_nbo_out;
  float* d_cbo_out;
  cudaMalloc((void**)&d_vbo_out, numVoxels * m_cube.vbosize * sizeof(float));
  cudaMalloc((void**)&d_ibo_out, numVoxels * m_cube.ibosize * sizeof(int));
  cudaMalloc((void**)&d_nbo_out, numVoxels * m_cube.nbosize * sizeof(float));
  cudaMalloc((void**)&d_cbo_out, numVoxels * m_cube.nbosize * sizeof(float));

  //Warn if vbo and nbo are not same size on cube
  if (m_cube.vbosize != m_cube.nbosize) {
    std::cout << "ERROR: cube vbo and nbo have different sizes." << std::endl;
    return;
  }

  //Create resulting cube-ized mesh
  createCubeMesh<<<(numVoxels / 256) + 1, 256>>>(d_voxels, d_values, M, T, bbox0, t_d, p_d, vox_size / CUBE_MESH_SCALE, numVoxels, thrust::raw_pointer_cast(&d_vbo_cube.front()),
    m_cube.vbosize, thrust::raw_pointer_cast(&d_ibo_cube.front()), m_cube.ibosize, thrust::raw_pointer_cast(&d_nbo_cube.front()), d_vbo_out, d_ibo_out, d_nbo_out, d_cbo_out);

  //Store output sizes
  m_out.vbosize = numVoxels * m_cube.vbosize;
  m_out.ibosize = numVoxels * m_cube.ibosize;
  m_out.nbosize = numVoxels * m_cube.nbosize;
  m_out.cbosize = m_out.nbosize;

  //Memory allocation for the outputs
  m_out.vbo = (float*)malloc(m_out.vbosize * sizeof(float));
  m_out.ibo = (int*)malloc(m_out.ibosize * sizeof(int));
  m_out.nbo = (float*)malloc(m_out.nbosize * sizeof(float));
  m_out.cbo = (float*)malloc(m_out.cbosize * sizeof(float));

  //Sync here after doing some CPU work
  cudaDeviceSynchronize();

  //Copy data back from GPU
  //TODO: Can we avoid this step by making everything run from device-side VBO/IBO/NBO/CBO?
  cudaMemcpy(m_out.vbo, d_vbo_out, m_out.vbosize*sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(m_out.ibo, d_ibo_out, m_out.ibosize*sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(m_out.nbo, d_nbo_out, m_out.nbosize*sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(m_out.cbo, d_cbo_out, m_out.cbosize*sizeof(float), cudaMemcpyDeviceToHost);

  ///Free GPU memory
  cudaFree(d_vbo_out);
  cudaFree(d_ibo_out);
  cudaFree(d_nbo_out);
  cudaFree(d_cbo_out);

}

__host__ void voxelizeToCubes(Mesh &m_in, Mesh &m_cube, Mesh &m_out) {
  
  //Voxelize the mesh input
  int numVoxels = N*N*N;
  int* d_voxels;
  int* d_values;
  cudaMalloc((void**)&d_voxels, numVoxels*sizeof(int));
  cudaMalloc((void**)&d_values, numVoxels*sizeof(int));
  numVoxels = voxelizeMesh(m_in, d_voxels, d_values);

  //Extract Cubes from the Voxel Grid
  extractCubesFromVoxelGrid(d_voxels, numVoxels, d_values, m_cube, m_out);

  cudaFree(d_voxels);
  cudaFree(d_values);
}

