A Voxel Rendering Pipeline in CUDA for Real-time Indirect Illumination
======================

Dave Kotfis and Jiawei Wang
CIS 565 Final Project � Fall 2014

Goal - Implement a voxel rendering pipeline in CUDA capable of real-time global illumination.

Approach
- Utilize open source VoxelPipe library from NVidia to get up and running quickly.
- Reuse our CUDA Rasterizer to visualize the result and check our progress.
- Build a Sparse Voxel Octree representation on the GPU.
- Implement octree cone tracing for indirect illumination by Crassin et. al.
- Compare performance against our CUDA Pathtracer.


![Project Plan] (images/project_plan.png)

##Phase I Progress:

- Completed OpenGL Equivalent Rasterization Pipeline with quick swap to CUDA Rasterizer.
- Integrated NVidia VoxelPipe
- Added conversion kernels that can create a cube-ized mesh to render through the standard rasterizers (CUDA pipeline does not seem to be working yet for unknown reasons).

![Voxel Dragon] (images/stanford_dragon_voxelized_gl.png)

![Voxel Bunny] (images/bunny_voxelized_gl.png)
