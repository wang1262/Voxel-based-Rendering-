/*
 * Copyright (c) 2010-2011, NVIDIA Corporation
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above copyright
 *     notice, this list of conditions and the following disclaimer in the
 *     documentation and/or other materials provided with the distribution.
 *   * Neither the name of NVIDIA Corporation nor the
 *     names of its contributors may be used to endorse or promote products
 *     derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <voxelizer_test.h>
#include <glm/glm.h>
#include <GL/glut.h>

#include <nih/basic/stats.h>
#include <nih/time/timer.h>
#include <nih/linalg/matrix.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>

#include <voxelpipe/voxelpipe.h>

using namespace nih;

void test_voxelizer(const char* filename)
{
    fprintf(stderr, "test_voxelizer \"%s\"\n", filename);

	GLMmodel* model = glmReadOBJ( filename );

	glmUnitize( model );

    const int32 n_triangles = model->numtriangles;
    const int32 n_vertices  = model->numvertices;

    thrust::host_vector<int4>   h_triangles( n_triangles );
    thrust::host_vector<float4> h_vertices( n_vertices );

    fprintf(stderr, "  model loaded: %d triangles\n", n_triangles);

    const float* vertices = model->vertices + 3; // GLM offsets vertices by 1

    for (int32 i = 0; i < n_vertices; i++)
    {
        h_vertices[i].x = vertices[i*3 + 0];
        h_vertices[i].y = vertices[i*3 + 1];
        h_vertices[i].z = vertices[i*3 + 2];
    }

    for (int32 i = 0; i < n_triangles; i++)
	{
		h_triangles[ i ].x = model->triangles[i].vindices[0]-1;
		h_triangles[ i ].y = model->triangles[i].vindices[1]-1;
		h_triangles[ i ].z = model->triangles[i].vindices[2]-1;
	}

    const float3 bbox0 = make_float3( -1.1f, -1.1f, -1.1f );
    const float3 bbox1 = make_float3(  1.1f,  1.1f,  1.1f );

    thrust::device_vector<int4>        d_triangles( h_triangles );
    thrust::device_vector<float4>      d_vertices( h_vertices );

    const int32 N_tests = 10;

    //const int32 log_N = 9;
    //const int32 log_T = 5;
    const int32 log_N = 7;
    const int32 log_T = 4;
    //const int32 log_N = 10;
    //const int32 log_T = 5;
    //const int32 log_N = 9;
    //const int32 log_T = 5;

    // regular voxelization
    {
        voxelpipe::FRContext<log_N,log_T>  context;

        context.reserve( n_triangles, 1024u*1024u*16u );

        typedef voxelpipe::FR::TileOp<voxelpipe::Bit,voxelpipe::ADD_BLENDING,log_T> tile_op_type;
        typedef typename tile_op_type::storage_type storage_type;

        const int32 M = 1 << (log_N - log_T);
        thrust::device_vector<uint8>  d_fb( M*M*M * sizeof( storage_type ) * tile_op_type::STORAGE_SIZE );

        Timer timer;
        timer.start();

        for (int32 i = 0; i < N_tests; ++i)
        {
            context.coarse_raster(
                n_triangles,
                n_vertices,
                thrust::raw_pointer_cast( &d_triangles.front() ),
                thrust::raw_pointer_cast( &d_vertices.front() ),
                bbox0,
                bbox1 );

            context.fine_raster< voxelpipe::Bit, voxelpipe::BIT_FORMAT, voxelpipe::THIN_RASTER, voxelpipe::ADD_BLENDING, voxelpipe::DefaultShader<voxelpipe::Bit> >(
                n_triangles,
                n_vertices,
                thrust::raw_pointer_cast( &d_triangles.front() ),
                thrust::raw_pointer_cast( &d_vertices.front() ),
                bbox0,
                bbox1,
                thrust::raw_pointer_cast( &d_fb.front() ) );
        }

        timer.stop();

        const float total_time = timer.seconds();

        fprintf(stderr, "  test [%d,%d]... done\n", log_N, log_T);

        fprintf(stderr, "  FR:\n" );
        fprintf(stderr, "    coarse raster : %.3f ms\n", context.coarse_time * 1000.0f / N_tests );
        fprintf(stderr, "      sorting     : %.3f ms\n", context.sorting_time * 1000.0f / N_tests );
        fprintf(stderr, "      raster      : %.3f ms\n", (context.coarse_time - context.sorting_time) * 1000.0f / N_tests );
        fprintf(stderr, "    fine raster   : %.3f ms\n", context.fine_time  * 1000.0f / N_tests );
        fprintf(stderr, "      splitting   : %.3f ms\n", context.splitting_time * 1000.0f / N_tests );
        fprintf(stderr, "      blending    : %.3f ms\n", context.blend_time * 1000.0f / N_tests );
        fprintf(stderr, "      raster      : %.3f ms\n", (context.fine_time - context.splitting_time - context.blend_time) * 1000.0f / N_tests );
        fprintf(stderr, "    total         : %.3f ms\n", total_time * 1000.0f / N_tests );
        fprintf(stderr, "    tris/sec      : %.3f M\n", float(N_tests) * (float(n_triangles) / (timer.seconds() * 1.0e6f)) );

        #if VOXELPIPE_ENABLE_PROFILING
        fprintf(stderr, "  stats:\n");
        fprintf(stderr, "    fragment count    : %u\n", context.fr_stats.fragment_count );
        fprintf(stderr, "    FR utilization(tile test)   : %.2f\n", context.fr_stats.utilization[0] );
        fprintf(stderr, "    FR utilization(sample test) : %.2f\n", context.fr_stats.utilization[1] );
        fprintf(stderr, "    FR samples/tri              : %.2f, std: %.2f\n", context.fr_stats.samples_avg, context.fr_stats.samples_std );
        #endif
    }
    // A-buffer
    {
        voxelpipe::ABufferContext<log_N>  context;

        context.reserve( n_triangles, 1024u*1024u*32u );

        Timer timer;
        timer.start();

        for (int32 i = 0; i < N_tests; ++i)
        {
            context.run(
                n_triangles,
                n_vertices,
                thrust::raw_pointer_cast( &d_triangles.front() ),
                thrust::raw_pointer_cast( &d_vertices.front() ),
                bbox0,
                bbox1 );
        }

        timer.stop();

        const float total_time = timer.seconds();

        fprintf(stderr, "  AB:\n" );
        fprintf(stderr, "    setup       : %.3f ms\n", context.setup_time * 1000.0f / N_tests );
        fprintf(stderr, "    emit        : %.3f ms\n", context.emit_time * 1000.0f / N_tests );
        fprintf(stderr, "    sorting     : %.3f ms\n", context.sorting_time * 1000.0f / N_tests );
        fprintf(stderr, "    total         : %.3f ms\n", total_time * 1000.0f / N_tests );
        fprintf(stderr, "    tris/sec      : %.3f M\n", float(N_tests) * (float(n_triangles) / (timer.seconds() * 1.0e6f)) );
    }
}
