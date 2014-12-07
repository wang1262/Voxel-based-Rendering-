A Voxel Rendering Pipeline in CUDA for Real-time Indirect Illumination
======================

See our results on [YouTube](http://youtu.be/UgqYFQuIGmc)

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

##Phase III Progress

- Texture Mapping in VoxelPipe is Complete
- SVO Cube Extraction is Complete
- Filling values to interior SVO nodes is Complete
- Added the ability to extract cubes from SVO at arbitrary resolution

This shows the stanford dragon with textures mapped into the voxel grid, then filled into the SVO and extracted.

![Textured Dragon Octree] (images/textured_dragon_svo_gl.png)

This is the same as before, though the cubes rendered are extracted from a higher level in the octree than the original voxelization, so the values are mip-mapped.

![Textured Dragon Octree Mip-Mapped] (images/textured_dragon_svo_mipmapped_gl.png)

##Performance Analysis

Data Structure Sizes:

| Model | Resolution | # Occupied Voxels | # SVO Nodes |
| --- | --- | --- | --- |
| Bunny | 128 | 3,234 | 9,656 |
|  | 256 | 11,603 | 38,408 |
|  | 512 | 43,126 | 151,552 |
| Dragon | 128 | 6,327 | 14,696 |
|  | 256 | 20,604 | 58,536 |
|  | 512 | 69,865 | 225,360 |

The following timing statistics were produced using an NVidia GTX 770.
Data Construction / Cube Extraction Timing:

| Model | Resolution | Voxelization | Voxel Grid to Cubes | SVO from Voxel Grid | SVO to Cubes |
| --- | --- | --- | --- | --- | --- |
| Bunny | 128 | 16.2 ms | 8.46 ms | 1.85 ms | 7.78 ms |
|  | 256 | 54.2 ms | 19.7 ms | 2.07 ms | 22.6 ms |
|  | 512 | 203.9 ms | 48.6 ms | 2.3 ms | 39.9 ms |
| Dragon | 128 | 26.8 ms | 12.1 ms | 1.06 ms | 12.06 ms |
|  | 256 | 49 ms | 30.5 ms | 2.79 ms | 33.6 ms |
|  | 512 | 221 ms | 90.9 ms | 2.7 ms | 71.8 ms |

CUDA vs. OpenGL. Voxel rendering pipeline performance comparison.
(With NVidia GT750M)

| Voxelize | Model | CUDA Render Time | CUDA FPS | OpenGL Render Time | OpenGL FPS |
| --- | --- | --- | --- | --- | --- |
| Without Voxelization | Dragon | 47 ms | 18 | 15 ms | 60 |
|  | Two-Cows | 16 ms | 35 | 0~5 ms | 60 |
|  | All Three* | 55 ms | 17 | 15 ms | 58 |
|  | Bunny | 31 ms | 29 | ~0 ms | 60 |
| With Voxelization | Dragon | 156 ms | 5 | 31 ms | 30 |
|  | Two-Cows | 23 ms | 27 | ~0 ms | 60 |
|  | All Three | 141 ms | 7 | 16 ms | 34 |
|  | Bunny | 109 ms | 10 | 15 ms | 47 |

(*)All Three is a model scene contains dragon, bunny and buddha objects.

##References

[Interactive Indirect Illumination Using Voxel Cone Tracing] (https://hal.inria.fr/hal-00650196)

[Voxel Cone Tracing and Sparse Voxel Octree for Real-time Global Illumination](http://on-demand.gputechconf.com/gtc/2012/presentations/SB134-Voxel-Cone-Tracing-Octree-Real-Time-Illumination.pdf)

[OpenGL Insights - Octree-Based Sparse Voxelization Using the GPU Hardware Rasterizer] (http://www.seas.upenn.edu/~pcozzi/OpenGLInsights/OpenGLInsights-SparseVoxelization.pdf)

[GigaVoxels] (http://maverick.inria.fr/Membres/Cyril.Crassin/thesis/CCrassinThesis_EN_Web.pdf)

[VoxelPipe] (https://research.nvidia.com/sites/default/files/publications/voxel-pipe.pdf)

##Bloopers

This is what happens when not enough memory is allocated for the SVO.

![2 Cows No Memory SVO] (images/2_cows_octree_out_of_memory.png)

![Dragon with Texture in GPU Memory] (images/dragon_octree_texture_map_512.png)

