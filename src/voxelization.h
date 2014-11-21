#ifndef VOXELIZATION_H
#define VOXELIZATION_H

#include <cuda.h>
#include "sceneStructs.h"

void voxelizeToCubes(Mesh &m_in, Mesh &m_cube, Mesh &m_out);

#endif ///VOXELIZATION_H
