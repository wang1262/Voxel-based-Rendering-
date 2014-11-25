
#include "voxelization.h"
#include <glm/glm.hpp>
#include <GL/glut.h>

#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>

#include <voxelpipe/voxelpipe.h>

//Declare voxelization resolution (TODO: input these as a parameter)
const int log_N = 8;
const int log_T = 4;

const float CUBE_MESH_SCALE = 0.1;

voxelpipe::FRContext<log_N, log_T>*  context;

bool first_time = true;

__device__ float3 getCenterFromIndex(int idx, int M, int T, float3 bbox0, float3 t_d, float3 p_d) {
  int T3 = T*T*T;
  int tile_num = idx / T3;
  int pix_num = idx % T3;
  float3 cent;
  int tx = tile_num / (M*M) % M;
  int px = pix_num / (T*T) % T;
  int ty = tile_num / M % M;
  int py = pix_num / T % T;
  int tz = tile_num % M;
  int pz = pix_num % T;
  cent.x = bbox0.x + tx*t_d.x + px*p_d.x;
  cent.y = bbox0.y + ty*t_d.y + py*p_d.y;
  cent.z = bbox0.z + tz*t_d.z + pz*p_d.z;
  return cent;
}

template<typename storage_type>
__global__ void getOccupiedVoxels(void* fb, int M, int T, int* voxels) {
  int T3 = T*T*T;
  int M3 = M*M*M;

  int pix_num = (blockIdx.x * 256 % T3) + threadIdx.x;
  int tile_num = blockIdx.x * 256 / T3;

  if (pix_num < T3 && tile_num < M3) {
    //TODO: Is there any benefit in making this shared?
    storage_type* tile;

    float3 cent;

    bool is_occupied;
    if (T <= 8) {
      tile = (storage_type*)fb + tile_num*T3;
      is_occupied = tile[pix_num];
    } else {
      tile = (storage_type*)fb + tile_num*(T3 >> 5);
      is_occupied = (tile[pix_num >> 5] & (1 << (pix_num & 31)));
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

__global__ void createCubeMesh(int* voxels, int M, int T, float3 bbox0, float3 t_d, float3 p_d, float scale_factor, int num_voxels, float* cube_vbo, 
                                int cube_vbosize, int* cube_ibo, int cube_ibosize, float* cube_nbo, float* out_vbo, int* out_ibo, float* out_nbo) {

  //Get the index for the thread
  int idx = blockIdx.x*blockDim.x + threadIdx.x;

  if (idx < num_voxels) {

    int vbo_offset = idx * cube_vbosize;
    int ibo_offset = idx * cube_ibosize;
    float3 center = getCenterFromIndex(voxels[idx], M, T, bbox0, t_d, p_d);

    for (int i = 0; i < cube_vbosize; i++) {
      if (i % 3 == 0) {
        out_vbo[vbo_offset + i] = cube_vbo[i] * scale_factor + center.x;
      } else if (i % 3 == 1) {
        out_vbo[vbo_offset + i] = cube_vbo[i] * scale_factor + center.y;
      } else {
        out_vbo[vbo_offset + i] = cube_vbo[i] * scale_factor + center.z;
      }
      out_nbo[vbo_offset + i] = cube_nbo[i];
    }

    for (int i = 0; i < cube_ibosize; i++) {
      out_ibo[ibo_offset + i] = cube_ibo[i] + ibo_offset;
    }

  }

}

void voxelizeToCubes(Mesh &m_in, Mesh &m_cube, Mesh &m_out) {
  
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

  const int N = 1 << log_N; //N is the total number of voxels (per dimension)
  const int M = 1 << (log_N - log_T); //M is the total number of tiles (per dimension)
  const int T = 1 << log_T; //T is the tile size - voxels per tile (per dimension)

  //Create bounding box to perform voxelization within
  const float world_size = 2.0f; //TODO: get this from the bounding box of the input mesh
  const float3 bbox0 = make_float3(-world_size, -world_size, -world_size);
  const float3 bbox1 = make_float3(world_size, world_size, world_size);

  //Compute the 1/2 edge length for the resulting voxelization
  const float vox_size = world_size / float(N);

  //Compute tile/grid sizes
  float3 t_d = make_float3((bbox1.x - bbox0.x) / float(M),
    (bbox1.y - bbox0.y) / float(M),
    (bbox1.z - bbox0.z) / float(M));
  float3 p_d = make_float3(t_d.x / float(T),
    t_d.y / float(T), t_d.z / float(T));

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

  //Perform coarse and fine voxelization
  context->coarse_raster(n_triangles, n_vertices, thrust::raw_pointer_cast(&d_triangles.front()), thrust::raw_pointer_cast(&d_vertices.front()), bbox0, bbox1);
  context->fine_raster< voxelpipe::Bit, voxelpipe::BIT_FORMAT, voxelpipe::THIN_RASTER, voxelpipe::ADD_BLENDING, voxelpipe::DefaultShader<voxelpipe::Bit> >(
      n_triangles, n_vertices, thrust::raw_pointer_cast(&d_triangles.front()), thrust::raw_pointer_cast(&d_vertices.front()), bbox0, bbox1, thrust::raw_pointer_cast(&d_fb.front()));
  
  //TODO: Consider replacing this next part with scan/compact/decode rather than decode/compact

  //Get voxel centers
  int numVoxels = N*N*N;
  thrust::device_vector<int> d_vox(numVoxels);
  getOccupiedVoxels<storage_type> <<<N*N*N, 256>>>(thrust::raw_pointer_cast(&d_fb.front()), M, T, thrust::raw_pointer_cast(&d_vox.front()));
  cudaDeviceSynchronize();

  //Stream Compact voxels to remove the empties
  thrust::device_vector<int> d_vox2(numVoxels);
  numVoxels = thrust::copy_if(&d_vox.front(), &d_vox.front() + numVoxels, &d_vox2.front(), check_voxel()) - &d_vox2.front();

  //Move cube data to GPU
  thrust::device_vector<float> d_vbo_cube(m_cube.vbo, m_cube.vbo + m_cube.vbosize);
  thrust::device_vector<int> d_ibo_cube(m_cube.ibo, m_cube.ibo + m_cube.ibosize);
  thrust::device_vector<float> d_nbo_cube(m_cube.nbo, m_cube.nbo + m_cube.nbosize);

  //Create output structs
  float* d_vbo_out;
  int* d_ibo_out;
  float* d_nbo_out;
  cudaMalloc((void**)&d_vbo_out, numVoxels * m_cube.vbosize * sizeof(float));
  cudaMalloc((void**)&d_ibo_out, numVoxels * m_cube.ibosize * sizeof(int));
  cudaMalloc((void**)&d_nbo_out, numVoxels * m_cube.nbosize * sizeof(float));

  //Warn if vbo and nbo are not same size on cube
  if (m_cube.vbosize != m_cube.nbosize) {
    std::cout << "ERROR: cube vbo and nbo have different sizes." << std::endl;
    return;
  }

  //Create resulting cube-ized mesh
  createCubeMesh << <(numVoxels / 256) + 1, 256 >> >(thrust::raw_pointer_cast(&d_vox2.front()), M, T, bbox0, t_d, p_d, vox_size / CUBE_MESH_SCALE, numVoxels, thrust::raw_pointer_cast(&d_vbo_cube.front()),
    m_cube.vbosize, thrust::raw_pointer_cast(&d_ibo_cube.front()), m_cube.ibosize, thrust::raw_pointer_cast(&d_nbo_cube.front()), d_vbo_out, d_ibo_out, d_nbo_out);

  //Store output sizes
  m_out.vbosize = numVoxels * m_cube.vbosize;
  m_out.ibosize = numVoxels * m_cube.ibosize;
  m_out.nbosize = numVoxels * m_cube.nbosize;

  //Memory allocation for the outputs
  m_out.vbo = (float*)malloc(m_out.vbosize * sizeof(float));
  m_out.ibo = (int*)malloc(m_out.ibosize * sizeof(int));
  m_out.nbo = (float*)malloc(m_out.nbosize * sizeof(float));

  //Copy CBO from input directly
  m_out.cbo = (float*) malloc(m_in.cbosize * sizeof(float));
  memcpy(m_out.cbo, m_in.cbo, m_in.cbosize * sizeof(float));
  m_out.cbosize = m_in.cbosize;

  //Sync here after doing some CPU work
  cudaDeviceSynchronize();

  //Copy data back from GPU
  //TODO: Can we avoid this step by making everything run from device-side VBO/IBO/NBO/CBO?
  cudaMemcpy(m_out.vbo, d_vbo_out, m_out.vbosize*sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(m_out.ibo, d_ibo_out, m_out.ibosize*sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(m_out.nbo, d_nbo_out, m_out.nbosize*sizeof(float), cudaMemcpyDeviceToHost);

  ///Free GPU memory
  cudaFree(d_vbo_out);
  cudaFree(d_ibo_out);
  cudaFree(d_nbo_out);
}

