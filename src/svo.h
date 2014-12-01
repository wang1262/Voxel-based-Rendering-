#ifndef SVO_H
#define SVO_H

#include <cuda.h>
#include "sceneStructs.h"

__host__ void svoFromVoxels(int* d_voxels, int numVoxels, int* d_octree);

__host__ void extractCubesFromSVO(int* d_octree, Mesh &m_cube, Mesh &m_out);

__host__ void voxelizeSVOCubes(Mesh &m_in, Mesh &m_cube, Mesh &m_out);

#endif ///SVO_H
