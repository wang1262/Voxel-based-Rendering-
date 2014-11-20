#ifndef VOXELIZATION_H
#define VOXELIZATION_H

#include <cuda.h>

void voxelize(float* vbo, int vbosize, float* cbo, int cbosize, int* ibo, int ibosize, float* nbo, int nbosize);

#endif ///VOXELIZATION_H
