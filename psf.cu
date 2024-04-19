
#include "tira/cuda/error.h"
#include "tira/optics/planewave.h"
#include <cuda/std/complex>
using namespace cuda::std;

__global__ void kernelPSF(complex<float>* field, unsigned int field_res, float extent, float w, int axis,
    complex<float>* aperture, float sin_alpha, unsigned int fa_res, float refractive_index, float lambda) {
    unsigned int ui = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int vi = blockIdx.y * blockDim.y + threadIdx.y;

    if (ui >= field_res || vi >= field_res) return;

    unsigned int N = field_res;
    float d = (2 * extent) / (N - 1);
    float u = -extent + d * ui;
    float v = -extent + d * vi;

    float ds = (2 * sin_alpha) / (fa_res - 1);

    float k_mag = 6.283185307179586 / lambda * refractive_index;
    tira::vec3<float> s;
    tira::cvec3<float> E(0.0f, 0.0f, 0.0f);
    for (unsigned int syi = 0; syi < fa_res; syi++) {
        s[1] = -sin_alpha + ds * syi;
        for (unsigned int sxi = 0; sxi < fa_res; sxi++) {
            s[0] = -sin_alpha + ds * sxi;

            float sx2_sy2 = s[0] * s[0] + s[1] * s[1];
            if (sx2_sy2 <= 1) {

                s[2] = sqrt(1 - sx2_sy2);
                tira::vec3<float> k(k_mag * s[0], k_mag * s[1], k_mag * s[2]);
                size_t i = syi * fa_res * 3 + sxi * 3;
                tira::cvec3<float> A(aperture[i + 0], aperture[i + 1], aperture[i + 2]);
                tira::planewave<float> p(k[0], k[1], k[2], A[0], A[1], A[2]);
                if (axis == 0)
                    E += p.E(w, u, v);
                else if (axis == 1)
                    E += p.E(u, w, v);
                else
                    E += p.E(u, v, w);
            }
        }
    }
    size_t i = vi * field_res * 3 + ui * 3;
    field[i + 0] = E[0];
    field[i + 1] = E[1];
    field[i + 2] = E[2];
}

__global__ void kernelPSFSubstrate(complex<float>* field, unsigned int field_res, float extent, float w, int axis, complex<float> nr,
    complex<float>* aperture, float sin_alpha, unsigned int fa_res, float refractive_index, float lambda) {
    unsigned int ui = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int vi = blockIdx.y * blockDim.y + threadIdx.y;

    if (ui >= field_res || vi >= field_res) return;

    unsigned int N = field_res;
    float d = (2 * extent) / (N - 1);
    float u = -extent + d * ui;
    float v = -extent + d * vi;

    float ds = (2 * sin_alpha) / (fa_res - 1);

    float k_mag = 6.283185307179586 / lambda * refractive_index;
    tira::vec3<float> s;
    tira::cvec3<float> E(0.0f, 0.0f, 0.0f);
    for (unsigned int syi = 0; syi < fa_res; syi++) {
        s[1] = -sin_alpha + ds * syi;
        for (unsigned int sxi = 0; sxi < fa_res; sxi++) {
            s[0] = -sin_alpha + ds * sxi;

            float sx2_sy2 = s[0] * s[0] + s[1] * s[1];
            if (sx2_sy2 <= 1) {

                s[2] = sqrt(1 - sx2_sy2);
                tira::vec3<float> k(k_mag * s[0], k_mag * s[1], k_mag * s[2]);
                size_t i = syi * fa_res * 3 + sxi * 3;
                tira::cvec3<float> A(aperture[i + 0], aperture[i + 1], aperture[i + 2]);
                tira::planewave<float> p(k[0], k[1], k[2], A[0], A[1], A[2]);
                if (axis == 0) {
                    if (v <= 0) {
                        E += p.E(w, u, v);
                        E += p.reflect(nr).E(w, u, v);
                    }
                    else {
                        //E += p.refract(nr).E(w, u, v);
                    }
                }
                else if (axis == 1) {
                    if (v <= 0) {
                        E += p.E(u, w, v);
                        //E += p.reflect(nr).E(u, w, v);
                    }
                    else {
                        //E += p.refract(nr).E(u, w, v);
                    }
                }
                else {
                    if (w <= 0) {
                        E += p.E(u, v, w);
                        //E += p.reflect(nr).E(u, v, w);
                    }
                    else {
                        //E += p.refract(nr).E(u, w, v);
                    }
                }
            }
        }
    }
    size_t i = vi * field_res * 3 + ui * 3;
    field[i + 0] = E[0];
    field[i + 1] = E[1];
    field[i + 2] = E[2];
}

__global__ void kernelTest(complex<float>* field, unsigned int field_res, float extent, float w, int axis,
    complex<float>* aperture, float sin_alpha, unsigned int fa_res, float refractive_index, float lambda) {
    unsigned int ui = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int vi = blockIdx.y * blockDim.y + threadIdx.y;

    if (ui >= field_res || vi >= field_res) return;

    field[vi * field_res * 3 + ui * 3 + 0] = ui;
    field[vi * field_res * 3 + ui * 3 + 1] = vi;
    field[vi * field_res * 3 + ui * 3 + 2] = ui * vi;
}

void gpuPSF(complex<float>* field, unsigned int field_res, float extent, float w, int axis,
    complex<float>* aperture, float sin_alpha, unsigned int fa_res, float refractive_index, float lambda, int device) {

    complex<float>* gpu_field;
    unsigned int field_size = sizeof(complex<float>) * field_res * field_res * 3;
    HANDLE_ERROR(cudaMalloc(&gpu_field, field_size));

    complex<float>* gpu_aperture;
    unsigned int aperture_size = sizeof(complex<float>) * fa_res * fa_res * 3;
    HANDLE_ERROR(cudaMalloc(&gpu_aperture, aperture_size));
    HANDLE_ERROR(cudaMemcpy(gpu_aperture, aperture, aperture_size, cudaMemcpyHostToDevice));

    dim3 threads(32, 32);
    dim3 blocks(field_res / threads.x + 1, field_res / threads.y + 1);
    kernelPSF <<<blocks, threads >>> (gpu_field, field_res, extent, w, axis, gpu_aperture, sin_alpha, fa_res, refractive_index, lambda);

    HANDLE_ERROR(cudaMemcpy(field, gpu_field, field_size, cudaMemcpyDeviceToHost));
}

void gpuPSFSubstrate(complex<float>* field, unsigned int field_res, float extent, float w, int axis, complex<float> nr,
    complex<float>* aperture, float sin_alpha, unsigned int fa_res, float refractive_index, float lambda, int device) {

    complex<float>* gpu_field;
    unsigned int field_size = sizeof(complex<float>) * field_res * field_res * 3;
    HANDLE_ERROR(cudaMalloc(&gpu_field, field_size));

    complex<float>* gpu_aperture;
    unsigned int aperture_size = sizeof(complex<float>) * fa_res * fa_res * 3;
    HANDLE_ERROR(cudaMalloc(&gpu_aperture, aperture_size));
    HANDLE_ERROR(cudaMemcpy(gpu_aperture, aperture, aperture_size, cudaMemcpyHostToDevice));

    dim3 threads(32, 32);
    dim3 blocks(field_res / threads.x + 1, field_res / threads.y + 1);
    kernelPSFSubstrate <<<blocks, threads >>> (gpu_field, field_res, extent, w, axis, nr, gpu_aperture, sin_alpha, fa_res, refractive_index, lambda);

    HANDLE_ERROR(cudaMemcpy(field, gpu_field, field_size, cudaMemcpyDeviceToHost));
}