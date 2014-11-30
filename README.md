A Voxel Rendering Pipeline in CUDA for Real-time Indirect Illumination
======================

Dave Kotfis and Jiawei Wang
CIS 565 Final Project – Fall 2014

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
- Added conversion kernels that can create a cube-ized mesh to render through the standard rasterizers. Works with both OpenGL and CUDA rasterization.

OpenGL Stanford Dragon:
![Voxel Dragon GL] (images/stanford_dragon_voxelized_gl.png)

[OpenGL Pipeline] (https://www.youtube.com/watch?v=ynhujtzyh6s&list=UUpiGNsmZZFAvKHIT0y5Q32Q)

[CUDA Pipeline] (https://www.youtube.com/watch?v=cD8gOQchpoM&feature=youtu.be)

##Phase II Progress:

- Port to Linux
- Texture Mapping (in progress)
- Increased efficiency for 1024^3 voxels
- GPU SVO (in progress)

[1024^3 Voxels Stanford Dragon](https://www.youtube.com/watch?v=wFguF1bXP6g&feature=youtu.be)

Building Voxel Octree
---------------------
Reference: [Simple Octree](https://github.com/brandonpelfrey/SimpleOctree)
