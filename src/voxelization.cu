
#include "voxelization.h"
#include <glm/glm.hpp>
#include <GL/glut.h>

#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>

#include <voxelpipe/voxelpipe.h>

void voxelize(float* vbo, int vbosize, float* cbo, int cbosize, int* ibo, int ibosize, float* nbo, int nbosize) {
  
  //Initialize sizes
  const int n_triangles = ibosize / 3;
  const int n_vertices = vbosize / 3;

  //Create host vectors
  thrust::host_vector<int4> h_triangles(n_triangles);
  thrust::host_vector<float4> h_vertices(n_vertices);

  //Fill in the data
  for (int i = 0; i < n_vertices; i++) {
    h_vertices[i].x = vbo[i * 3 + 0];
    h_vertices[i].y = vbo[i * 3 + 1];
    h_vertices[i].z = vbo[i * 3 + 2];
  }
  for (int i = 0; i < n_triangles; i++) {
    h_triangles[i].x = ibo[i * 3 + 0];
    h_triangles[i].y = ibo[i * 3 + 1];
    h_triangles[i].z = ibo[i * 3 + 2];
  }

  //Create bounding box to perform voxelization within
  const float3 bbox0 = make_float3(-1.1f, -1.1f, -1.1f);
  const float3 bbox1 = make_float3(1.1f, 1.1f, 1.1f);

  //Copy to device vectors
  thrust::device_vector<int4> d_triangles(h_triangles);
  thrust::device_vector<float4> d_vertices(h_vertices);

  //Declar voxelization resolution (TODO: input these as a parameter)
  const int log_N = 7;
  const int log_T = 4;

  //Create voxelpipe context
  voxelpipe::FRContext<log_N, log_T>  context;

  //Reserve data for voxelpipe
  context.reserve(n_triangles, 1024u * 1024u * 16u);

  //Define types for the voxelization
  typedef voxelpipe::FR::TileOp<voxelpipe::Bit, voxelpipe::ADD_BLENDING, log_T> tile_op_type;
  typedef typename tile_op_type::storage_type storage_type;

  const int M = 1 << (log_N - log_T);

  //Initialize the result data on the device
  thrust::device_vector<uint8_t>  d_fb(M*M*M * sizeof(storage_type) * tile_op_type::STORAGE_SIZE);

  //Perform coarse and fine voxelization
  context.coarse_raster(n_triangles, n_vertices, thrust::raw_pointer_cast(&d_triangles.front()), thrust::raw_pointer_cast(&d_vertices.front()), bbox0, bbox1);
  context.fine_raster< voxelpipe::Bit, voxelpipe::BIT_FORMAT, voxelpipe::THIN_RASTER, voxelpipe::ADD_BLENDING, voxelpipe::DefaultShader<voxelpipe::Bit> >(
      n_triangles, n_vertices, thrust::raw_pointer_cast(&d_triangles.front()), thrust::raw_pointer_cast(&d_vertices.front()), bbox0, bbox1, thrust::raw_pointer_cast(&d_fb.front()));
      
}
