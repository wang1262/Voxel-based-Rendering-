
#include "svo.h"
#include "voxelization.h"

__global__ void flagNodes(int* voxels, int numVoxels, int* octree, int M, int T, float3 bbox0, float3 t_d, float3 p_d, int tree_depth) {

  int index = blockIdx.x * blockDim.x + threadIdx.x;

  //Don't do anything if its out of bounds
  if (index < numVoxels) {
    float3 center = getCenterFromIndex(voxels[index], M, T, bbox0, t_d, p_d);
    float edge_length = abs(bbox0.x);
    float3 center_depth = make_float3(0.0f, 0.0f, 0.0f);
    int node_idx = 0;
    int this_node;

    //Loop until the specified depth
    for (int i = 0; i < tree_depth; i++) {
      int x = center.x > center_depth.x;
      int y = center.y > center_depth.y;
      int z = center.z > center_depth.z;
      this_node = node_idx + (x + (y << 1) + (z << 2));

      if (i < tree_depth - 1) {
        //The lowest 30 bits are the address
        node_idx = octree[2*this_node] & 0x3FFFFFFF;

        //Update the center depth for the next iteration
        center_depth.x += edge_length / 2 * (x ? 1 : -1);
        center_depth.y += edge_length / 2 * (y ? 1 : -1);
        center_depth.z += edge_length / 2 * (z ? 1 : -1);
      }
      edge_length /= 2.0f;
    }
    octree[2*this_node] = octree[2*this_node] | 0x40000000;
  }

}

__global__ void splitNodes(int* octree, int* numNodes, int poolSize) {

  int index = blockIdx.x * blockDim.x + threadIdx.x;

  //Don't do anything if its out of bounds
  if (index < poolSize) {
    int node = octree[2*index];

    //Split the node if its flagged
    if (node & 0x40000000) {
      //Get a new node tile
      int newNode = atomicAdd(numNodes, 8);

      //Point this node at the new tile
      octree[2 * index] = (octree[2 * index] & 0xC0000000) | (newNode & 0x3FFFFFFF);

      //Initialize new child nodes to 0's
      for (int off = 0; off < 8; off++) {
        octree[2*(newNode + off)] = 0;
      }
    }
  }

}

//This is based on Cyril Crassin's approach
__host__ void svoFromVoxels(int* d_voxels, int numVoxels, int* d_octree) {
  int numNodes = 8;
  int startingNode = 0;
  int* d_numNodes;
  cudaMalloc((void**)&d_numNodes, sizeof(int));
  cudaMemcpy(d_numNodes, &numNodes, sizeof(int), cudaMemcpyHostToDevice);
  int depth = 0;

  while (numNodes < (numVoxels*log_N) && ++depth < log_N) {

    //First, parallelize on voxels and flag nodes to be subdivided
    flagNodes<<<(numVoxels / 256) + 1, 256>>>(d_voxels, numVoxels, d_octree, M, T, bbox0, t_d, p_d, depth);

    cudaDeviceSynchronize();

    //Then, parallize on nodes and subdivide
    splitNodes<<<((numNodes - startingNode) / 256) + 1, 256>>>(&d_octree[2*startingNode], d_numNodes, numNodes - startingNode);
    startingNode = numNodes;

    cudaDeviceSynchronize();
    cudaMemcpy(&numNodes, d_numNodes, sizeof(int), cudaMemcpyDeviceToHost);
  }

  cudaFree(d_numNodes);
}

__host__ void extractCubesFromSVO(int* d_octree, Mesh &m_cube, Mesh &m_out) {

}

__host__ void voxelizeSVOCubes(Mesh &m_in, Mesh &m_cube, Mesh &m_out) {

  //Voxelize the mesh input
  int numVoxels = N*N*N;
  int* d_voxels;
  cudaMalloc((void**)&d_voxels, numVoxels*sizeof(int));
  numVoxels = voxelizeMesh(m_in, d_voxels);

  //Create the octree
  int* d_octree = NULL;
  cudaMalloc((void**)&d_octree, 2*log_N*numVoxels*sizeof(int));
  svoFromVoxels(d_voxels, numVoxels, d_octree);

  //Extract cubes from the leaves of the octree
  //extractCubesFromSVO(d_octree, m_cube, m_out);

  /////TEMPORARY---------------
  //Extract Cubes from the Voxel Grid
  extractCubesFromVoxelGrid(d_voxels, numVoxels, m_cube, m_out);

  //Copy CBO from input directly
  m_out.cbo = (float*)malloc(m_in.cbosize * sizeof(float));
  memcpy(m_out.cbo, m_in.cbo, m_in.cbosize * sizeof(float));
  m_out.cbosize = m_in.cbosize;
  ////////----------------

  //Free up GPU memory
  cudaFree(d_voxels);
  cudaFree(d_octree);

}
