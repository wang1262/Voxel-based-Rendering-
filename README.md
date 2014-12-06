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
Two points of octree implementation:
 - Only pixels to be actually displayed need to be computed, with the actual screen resolution limiting the level of voxel detail required (limits the computational cost during rendering)
 
 - Interior voxels, fully enclosed voxels, are unnecessary to be included in the 3D data set (limits the amount of 3D voxel data, which is the storage space)

Reference: [Sparse voxel Octree](http://en.wikipedia.org/wiki/Sparse_voxel_octree), 
           [Simple Octree](https://github.com/brandonpelfrey/SimpleOctree)

##Phase III Progress

- Texture Mapping in VoxelPipe is Complete
- SVO Cube Extraction is Complete
- Filling values to interior SVO nodes is Complete

![Textured Dragon Octree] (images/textured_dragon_svo_gl.png)

##Bloopers

This is what happens when not enough memory is allocated for the SVO.

![2 Cows No Memory SVO] (images/2_cows_octree_out_of_memory.png)

![Dragon with Texture in GPU Memory] (images/dragon_octree_texture_map_512.png)

