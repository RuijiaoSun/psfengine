/// This is an OpenGL "Hello World!" file that provides simple examples of integrating ImGui with GLFW
/// for basic OpenGL applications. The file also includes headers for the TIRA::GraphicsGL classes, which
/// provide an basic OpenGL front-end for creating materials and models for rendering.


#include "tira/graphics_gl.h"
#include "tira/image.h"
#include "tira/image/colormap.h"
#include "tira/optics/planewave.h"

#include <GL/glew.h>
#include <GLFW/glfw3.h> // Will drag system OpenGL headers

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include "imgui_internal.h"
#include "ImGuiFileDialog/ImGuiFileDialog.h"
#include <extern/libnpy/npy.hpp>

#include <iostream>
#include <string>
#include <stdio.h>
#include <numbers>
#include <chrono>
#include <random>

typedef std::mt19937 TwisterRng;  // the Mersenne Twister with a popular choice of parameters
TwisterRng rng;                   // e.g. keep one global instance (per thread)



#include <cuda/std/complex>
#include "cuda_runtime.h"

void gpuPSF(cuda::std::complex<float>* field, unsigned int field_res, float extent, float w, int axis,
    cuda::std::complex<float>* aperture, float sin_alpha, unsigned int fa_res, float refractive_index, float lambda, int device);

void gpuPSFSubstrate(cuda::std::complex<float>* field, unsigned int field_res, float extent, float w, int axis, 
    tira::planewave<float>* I, tira::planewave<float>* R, tira::planewave<float>* T, unsigned int N,
    bool incident, bool reflected, bool transmitted);

// User interface variables
GLFWwindow* window;                                     // pointer to the GLFW window that will be created (used in GLFW calls to request properties)
double window_width = 1600;
double window_height = 1200;
const char* glsl_version = "#version 130";              // specify the version of GLSL
ImVec4 clear_color = ImVec4(0.0f, 0.0f, 0.0f, 1.00f);   // specify the OpenGL color used to clear the back buffer
float ui_scale = 1.5f;                                  // scale value for the UI and UI text

float a_slider_value;                                   // UI variable used to store the value of a floating point slider widget

// Global Settings
ColorMap cmapIntensity = ColorMap::Magma;
ColorMap cmapDiverging = ColorMap::BrewerBlk;
float item_width_scale = 0.95;
int device = 0;
int devices;

typedef float precision;                                // define the data type used for the precision
enum FieldDisplay {XReal, XImag, YReal, YImag, ZReal, ZImag, Intensity, IncoherentIntensity};
int ApertureExp = 6;                                    // exponent for the back aperture resolution (always a power of 2)
int ApertureRes = pow(2, ApertureExp) + 1;
int PSFExp = 6;                                         // exponent for the PSF resolution
int PSFRes = pow(2, PSFExp) + 1;
tira::image<std::complex<precision>> BackAperture;      // define the back aperture image
tira::glTexture* texBackAperture = NULL;
FieldDisplay dispBackAperture = Intensity;
precision gaussian_center[2] = { 0, 0 };
bool PSFIncident = true;
bool PSFReflected = true;
bool PSFTransmitted = true;

tira::image<std::complex<precision>> FrontAperture;     // define an image array for the front aperture
vector< tira::planewave<float> > Pi;
vector< tira::planewave<float> > Pr;
vector< tira::planewave<float> > Pt;
tira::glTexture* texFrontAperture = NULL;
FieldDisplay dispFrontAperture = Intensity;

tira::image<std::complex<precision>> PSF;               // image to display the point spread function
tira::glTexture* texPSF = NULL;
FieldDisplay dispPSF = XReal;

tira::image<precision> Incoherent;
unsigned int IncoherentCounter = 0;

precision lambda = 0.5;                                 // wavelength for the field
precision sin_alpha = 0.9;                              // sine of the angle subtended by the outer aperture of the objective
precision alpha = asin(sin_alpha);
precision obscuration = 0.2;                            // percentage of the objective taken up by the center obscuration
precision refractive_index = 1.0;                       // refractive index of the immersion medium
precision aperture_radius = 100;                        // radius of the back aperture (in wavelength units)
precision beam_W0 = 100;
std::complex<precision> polarization[2] = {1.0, 0.0};                 // set the polarization of the incoming light
precision psf_window_width = 6.0f;
precision psf_extent = psf_window_width / 2.0f;                             // extent of the PSF calculated (in wavelength units)
std::vector<tira::glTexture*> Textures(5);
precision planes[3] = { 0.0f, 0.0f, 0.0f };
int axis = 0;
precision spectrum[2] = { 0.2f, 0.7f };

double xpos, ypos;                                      // cursor positions
// UI Elements
precision display_size = 500;                             // size of each field image on the screen

bool substrate = true;
precision substrate_n_real = 1.4f;
precision substrate_n_imag = 0.0f;
precision psf_time;



/// Functions that assign values to the back aperture

// Fill the back aperture with a plane wave of polarization (xp, yp, 0)
void BA_PlaneWave(unsigned int resolution, std::complex<precision> xp, std::complex<precision> yp) {
    
    unsigned int N = resolution;
    BackAperture.resize({ N, N, 3 });
    for (unsigned int yi = 0; yi < N; yi++) {
        for (unsigned int xi = 0; xi < N; xi++) {
            BackAperture(xi, yi, 0) = xp;
            BackAperture(xi, yi, 1) = yp;
            BackAperture(xi, yi, 2) = 0;            
        }
    }
}

void BA_Gaussian(tira::image< complex<float> > &aperture, unsigned int resolution, std::complex<precision> xp, std::complex<precision> yp, precision width, precision cx, precision cy, precision lambda = 1, precision z = 0) {

    unsigned int N = resolution;
    aperture.resize({ N, N, 3 });
    precision x, y, y_sq;
    precision d = (2 * aperture_radius) / (N - 1);

    precision k = 2 * std::numbers::pi / lambda;
    precision z0 = std::numbers::pi / lambda * pow(width, 2);

    for (unsigned int yi = 0; yi < N; yi++) {
        y = -aperture_radius + d * yi - cy;
        y_sq = y * y;
        for (unsigned int xi = 0; xi < N; xi++) {
            x = -aperture_radius + d * xi - cx;
            precision rho_sq = y_sq + x * x;
            std::complex<precision> A0 = 1.0f / std::complex<precision>(0, z0);
            precision zeta = atan(z / z0);

            precision Wz = width * sqrt(1 + pow(z / z0, 2));
            precision U1 = (width / Wz);
            precision U2 = exp(-rho_sq / (Wz * Wz));
            std::complex<precision> U3;
            if (z == 0) {
                U3 = std::exp(std::complex<precision>(0, zeta - k * z));
            }
            else {
                precision Rz = z * (1 + pow(z0 / z, 2));
                U3 = std::exp(std::complex<precision>(0, zeta - k * z - k * (rho_sq / (2 * Rz))));
            }

            std::complex<precision> U = U1 * U2 * U3;
            aperture(xi, yi, 0) = xp * U;
            aperture(xi, yi, 1) = yp * U;
            aperture(xi, yi, 2) = 0;
        }
    }
}

// Optical functions that calculate the field at the front aperture of the objective
void FA(tira::image< complex<float> > &front, tira::image< complex<float> > &back, unsigned int resolution, precision max_sin_alpha, precision obscuration) {

    unsigned int N = resolution;
    precision sin_alpha = max_sin_alpha;
    precision dsa = (2 * sin_alpha) / (N - 1);

    front.resize({ N, N, 3 });
    for (unsigned int yi = 0; yi < N; yi++) {
        precision sy = -sin_alpha + dsa * yi;                                                         // get the y spatial frequency for the current row
        for (unsigned int xi = 0; xi < N; xi++) {
            precision sx = -sin_alpha + dsa * xi;                                                     // get the x spatial frequency for the current pixel
            precision sx2_sy2 = sx * sx + sy * sy;                                    // calculate the band for the current spatial frequency
            precision sx2_sy2_sqrt = sqrt(sx2_sy2);

            if (sx2_sy2_sqrt < sin_alpha && sx2_sy2_sqrt >= sin_alpha * obscuration) {
                precision sz = sqrt(1 - sx2_sy2);
                precision sqrt_sz_inv = 1.0 / sqrt(sz);
                precision vc = 1.0 / sx2_sy2_sqrt;

                precision C[3][3];
                if (isinf(vc)) {
                    C[0][0] = 1;
                    C[0][1] = 0;
                    C[0][2] = 0;
                    C[1][0] = 0;
                    C[1][1] = 1;
                    C[1][2] = 0;
                    C[2][0] = 0;
                    C[2][1] = 0;
                    C[2][2] = 0;
                }
                else {
                    precision vs[3] = { vc * (-sy), vc * sx, 0 };
                    precision vp[3] = { vc * (-sx), vc * (-sy), 0 };
                    precision vpp[3] = { vc * (-sx * sz), vc * (-sy * sz), vc * sx2_sy2 };

                    C[0][0] = vs[0] * vs[0] + vpp[0] * vp[0];
                    C[0][1] = vs[0] * vs[1] + vpp[0] * vp[1];
                    C[0][2] = vs[0] * vs[2] + vpp[0] * vp[2];
                    C[1][0] = vs[1] * vs[0] + vpp[1] * vp[0];
                    C[1][1] = vs[1] * vs[1] + vpp[1] * vp[1];
                    C[1][2] = vs[1] * vs[2] + vpp[1] * vp[2];
                    C[2][0] = vs[2] * vs[0] + vpp[2] * vp[0];
                    C[2][1] = vs[2] * vs[1] + vpp[2] * vp[1];
                    C[2][2] = vs[2] * vs[2] + vpp[2] * vp[2];
                }
                std::complex<precision> FA[3];
                FA[0] = sqrt_sz_inv * (C[0][0] * back(xi, yi, 0) + C[0][1] * back(xi, yi, 1) + C[0][2] * back(xi, yi, 2));
                FA[1] = sqrt_sz_inv * (C[1][0] * back(xi, yi, 0) + C[1][1] * back(xi, yi, 1) + C[1][2] * back(xi, yi, 2));
                FA[2] = sqrt_sz_inv * (C[2][0] * back(xi, yi, 0) + C[2][1] * back(xi, yi, 1) + C[2][2] * back(xi, yi, 2));
                front(xi, yi, 0) = FA[0];
                front(xi, yi, 1) = FA[1];
                front(xi, yi, 2) = FA[2];
            }
            else {
                front(xi, yi, 0) = 0;
                front(xi, yi, 1) = 0;
                front(xi, yi, 2) = 0;
            }
        }
    }    
}

// calculate the plane wave decomposition of the PSF
void PW(tira::image< complex<float> > &aperture, unsigned int fa_res, float l, std::vector< tira::planewave<float> > &I, std::vector< tira::planewave<float> > &R, std::vector< tira::planewave<float> > &T) {

    I.clear();
    R.clear();
    T.clear();

    // generate the incident, reflected, and refracted plane waves
    float ds = (2 * sin_alpha) / (fa_res - 1);
    complex<float> nr = complex<float>(substrate_n_real, substrate_n_imag) / refractive_index;

    float k_mag = 2 * std::numbers::pi / l * refractive_index;
    tira::vec3<float> s;

    for (unsigned int syi = 0; syi < fa_res; syi++) {
        s[1] = -sin_alpha + ds * syi;

        for (unsigned int sxi = 0; sxi < fa_res; sxi++) {
            s[0] = -sin_alpha + ds * sxi;

            float sx2_sy2 = s[0] * s[0] + s[1] * s[1];

            if (sx2_sy2 <= 1) {
                s[2] = sqrt(1 - sx2_sy2);
                size_t i = syi * fa_res + sxi;
                tira::cvec3<float> A(aperture(sxi, syi, 0), aperture(sxi, syi, 1), aperture(sxi, syi, 2));
                tira::vec3<float> k(k_mag * s[0], k_mag * s[1], k_mag * s[2]);
                float k_dot_r;
                tira::planewave<float> p(k[0], k[1], k[2], A[0], A[1], A[2]);

                // only add the plane wave it its power is bigger than zero
                if (p.I0() > 0) {
                    I.push_back(p);
                    if (substrate) {
                        tira::planewave<float> r, t;
                        p.scatter(nr, r, t);

                        R.push_back(r);
                        T.push_back(t);
                    }
                }
            }
        }
    }
}

void cpuPSF(complex<float>* field, unsigned int field_res, float extent, float w, int axis,
    complex<float>* aperture, float sin_alpha, unsigned int fa_res, float refractive_index, float lambda) {
    unsigned int N = field_res;
    float d = (2 * extent) / (N - 1);

    float ds = (2 * sin_alpha) / (fa_res - 1);

    float k_mag = 2 * std::numbers::pi / lambda * refractive_index;
    tira::vec3<float> s;

    for (unsigned int vi = 0; vi < N; vi++) {
        float v = -extent + d * vi;
        for (unsigned int ui = 0; ui < N; ui++) {
            float u = -extent + d * ui;

            tira::cvec3<float> E(0, 0, 0);
            for (unsigned int syi = 0; syi < fa_res; syi++) {
                s[1] = -sin_alpha + ds * syi;

                for (unsigned int sxi = 0; sxi < fa_res; sxi++) {
                    s[0] = -sin_alpha + ds * sxi;

                    float sx2_sy2 = s[0] * s[0] + s[1] * s[1];
                    if (sx2_sy2 <= 1) {

                        s[2] = sqrt(1 - sx2_sy2);
                        size_t i = syi * fa_res * 3 + sxi * 3;
                        tira::cvec3<float> A(aperture[i + 0], aperture[i + 1], aperture[i + 2]);
                        tira::vec3 k = { k_mag * s[0], k_mag * s[1], k_mag * s[2] };
                        tira::planewave<float> p(k[0], k[1], k[2], A[0], A[1], A[2]);
                        if (axis == 0) {
                            E += p.E(w, u, v);
                        }
                        else if (axis == 1) {
                            E += p.E(u, w, v);
                        }
                        else {
                            E += p.E(u, v, w);
                        }
                    }
                }
            }
            field[vi * field_res * 3 + ui * 3 + 0] = E[0];
            field[vi * field_res * 3 + ui * 3 + 1] = E[1];
            field[vi * field_res * 3 + ui * 3 + 2] = E[2];
        }
    }
}



void cpuPSFSubstrate(complex<float>* field, unsigned int field_res, float extent, float w, int axis, tira::planewave<float>* I, tira::planewave<float>* R, tira::planewave<float>* T, unsigned int N) {
    
    float d = (2 * extent) / (field_res - 1);

    for (unsigned int vi = 0; vi < field_res; vi++) {
        float v = -extent + d * vi;
        for (unsigned int ui = 0; ui < field_res; ui++) {
            float u = -extent + d * ui;

            tira::cvec3<float> E(0, 0, 0);
            for (unsigned int k = 0; k < N; k++) {


                if (axis == 0) {
                    if (v <= 0) {
                        E += I[k].E(w, u, v);
                        E += R[k].E(w, u, v);
                    }
                    else {
                        E += T[k].E(w, u, v);
                    }
                }
                else if (axis == 1) {
                    if (v <= 0) {
                        E += I[k].E(u, w, v);
                        E += R[k].E(u, w, v);
                    }
                    else {
                        E += T[k].E(u, w, v);
                    }
                }
                else {
                    if (w <= 0) {
                        E += I[k].E(u, v, w);
                        E += R[k].E(u, v, w);
                    }
                    else {
                        E += T[k].E(u, w, v);
                    }
                }
            }
            field[vi * field_res * 3 + ui * 3 + 0] = E[0];
            field[vi * field_res * 3 + ui * 3 + 1] = E[1];
            field[vi * field_res * 3 + ui * 3 + 2] = E[2];
        }
    }
}


tira::image<precision> getFieldIntensity(tira::image< std::complex<precision> > E) {
    unsigned int N = E.X();
    unsigned int M = E.Y();

    tira::image<precision> I(N, M);

    for (unsigned int yi = 0; yi < M; yi++) {
        for (unsigned int xi = 0; xi < N; xi++) {
            I(xi, yi) = std::norm(E(xi, yi, 0)) + std::norm(E(xi, yi, 1)) + std::norm(E(xi, yi, 2));
        }
    }
    return I;
}

tira::image<precision> getFieldMagnitude(tira::image< std::complex<precision> > E) {
    unsigned int N = E.X();
    unsigned int M = E.Y();

    tira::image<precision> Mag(N, M);

    for (unsigned int yi = 0; yi < M; yi++) {
        for (unsigned int xi = 0; xi < N; xi++) {
            Mag(xi, yi) = sqrt(std::norm(E(xi, yi, 0)) + std::norm(E(xi, yi, 1)) + std::norm(E(xi, yi, 2)));
        }
    }
    return Mag;
}

std::vector< tira::image< precision > > getComplexComponents(tira::image< std::complex<precision> > C) {
    unsigned int N = C.X();
    unsigned int M = C.Y();

    tira::image<precision> Real(N, M);
    tira::image<precision> Imag(N, M);

    for (unsigned int yi = 0; yi < M; yi++) {
        for (unsigned int xi = 0; xi < N; xi++) {
            Real(xi, yi) = std::real(C(xi, yi));
            Imag(xi, yi) = std::imag(C(xi, yi));
        }
    }

    return { Real, Imag };
}

void ResetIncoherent() {
    IncoherentCounter = 0;
    Incoherent = getFieldIntensity(PSF);
}

/// <summary>
/// Return a vector of color images corresponding to components of the vector field
/// </summary>
/// <param name="E">2D vector field to be mapped</param>
/// <returns></returns>
tira::image<unsigned char> ColormapField(tira::image< std::complex<precision> > E, FieldDisplay component) {
    tira::image<precision> scalar;
    precision maxabs;
    switch (component) {
    case XReal:
        scalar = getComplexComponents(E.channel(0))[0]; break;
    case XImag:
        scalar = getComplexComponents(E.channel(0))[1]; break;
    case YReal:
        scalar = getComplexComponents(E.channel(1))[0]; break;
    case YImag:
        scalar = getComplexComponents(E.channel(1))[1]; break;
    case ZReal:
        scalar = getComplexComponents(E.channel(2))[0]; break;
    case ZImag:
        scalar = getComplexComponents(E.channel(2))[1]; break;

    case Intensity:
        scalar = getFieldIntensity(E);
        maxabs = scalar.maxv();
        return scalar.cmap(0, maxabs, cmapIntensity);

    case IncoherentIntensity:
        //scalar = getFieldMagnitude(E);
        if (Incoherent.size() == 0) {
            ResetIncoherent();
        }
        maxabs = Incoherent.maxv();
        return Incoherent.cmap(0, maxabs, cmapIntensity);
    }

    maxabs = std::max(abs(scalar.minv()), abs(scalar.maxv()));
    return scalar.cmap(-maxabs, maxabs, cmapDiverging);
}

void CalculatePSF(tira::image< complex<precision> >& P, std::vector< tira::planewave<float> > &I, std::vector< tira::planewave<float> >& R, std::vector< tira::planewave<float> >& T) {
    auto start = std::chrono::high_resolution_clock::now();

    P.resize({ (size_t)PSFRes, (size_t)PSFRes, 3 });

    if (substrate) {
        if (device < 0) {
            cpuPSFSubstrate(P.data(), PSFRes, psf_extent, planes[axis], axis, I.data(), R.data(), T.data(), I.size());
        }
        else {
            gpuPSFSubstrate((cuda::std::complex<float>*)P.data(), PSFRes, psf_extent, planes[axis], axis,
                I.data(), R.data(), T.data(), I.size(),
                PSFIncident, PSFReflected, PSFTransmitted);
        }
    }
    else {
        if (device < 0)
            cpuPSF(P.data(), PSFRes, psf_extent, planes[axis], axis, FrontAperture.data(), sin_alpha, ApertureRes, refractive_index, lambda);
        else
            gpuPSF((cuda::std::complex<float>*)P.data(), PSFRes, psf_extent, planes[axis], axis, (cuda::std::complex<float>*)FrontAperture.data(), sin_alpha, ApertureRes, refractive_index, lambda, device);
    }

    auto end = std::chrono::high_resolution_clock::now();

    psf_time = std::chrono::duration<float>(end - start).count();
}



void UpdatePSFDisplay() {
    tira::image<unsigned char> color = ColormapField(PSF, dispPSF);
    if (texPSF == NULL) 
        texPSF = new tira::glTexture(color.data(), PSFRes, PSFRes, 0, GL_RGB, GL_RGB, GL_UNSIGNED_BYTE);
    else
        texPSF->AssignImage(color.data(), PSFRes, PSFRes, 0, GL_RGB, GL_RGB, GL_UNSIGNED_BYTE);

    texPSF->SetFilter(GL_NEAREST);
    
}

void UpdateIncoherent() {
    if (Incoherent.size() == 0) ResetIncoherent();
    
    IncoherentCounter++;
    std::uniform_real_distribution<float> lambda_distribution(spectrum[0], spectrum[1]);
    std::uniform_real_distribution<float> polarization_distribution(0, 2 * std::numbers::pi);

    float rand_lambda = lambda_distribution(rng);
    float rand_pol = polarization_distribution(rng);

    tira::image< complex<float> > random_back;
    BA_Gaussian(random_back, ApertureRes, cos(rand_pol), sin(rand_pol), beam_W0, gaussian_center[0], gaussian_center[1], rand_lambda);

    tira::image< complex<float> > random_front;
    FA(random_front, random_back, ApertureRes, sin_alpha, obscuration);

    std::vector< tira::planewave<float> > rand_I;
    std::vector< tira::planewave<float> > rand_R;
    std::vector< tira::planewave<float> > rand_T;
    PW(random_front, ApertureRes, rand_lambda, rand_I, rand_R, rand_T);               // calculate planewaves based on the randomized lambda

    tira::image< complex<precision> > rand_psf;
    CalculatePSF(rand_psf, rand_I, rand_R, rand_T);                     // calculate a PSF from the randomized wavelength and polarization

    tira::image< precision > rand_intensity = getFieldIntensity(rand_psf);
    Incoherent = Incoherent + rand_intensity;

    std::cout << "lambda: " << rand_lambda << std::endl;
    std::cout << "polarization: (" << cos(rand_pol) << ", " << sin(rand_pol) << ")" << std::endl;
    UpdatePSFDisplay();
}



void UpdatePSF() {

    CalculatePSF(PSF, Pi, Pr, Pt);
    
    if (dispPSF == IncoherentIntensity) ResetIncoherent();
    UpdatePSFDisplay();
}

void UpdatePlaneWaves() {
    PW(FrontAperture, ApertureRes, lambda, Pi, Pr, Pt);
    UpdatePSF();
}

void UpdateFADisplay() {
    tira::image<unsigned char> color = ColormapField(FrontAperture, dispFrontAperture);
    if (texFrontAperture == NULL)
        texFrontAperture = new tira::glTexture(color.data(), ApertureRes, ApertureRes, 0, GL_RGB, GL_RGB, GL_UNSIGNED_BYTE);
    else
        texFrontAperture->AssignImage(color.data(), ApertureRes, ApertureRes, 0, GL_RGB, GL_RGB, GL_UNSIGNED_BYTE);

    texFrontAperture->SetFilter(GL_NEAREST);

}
void UpdateFA() {
    FA(FrontAperture, BackAperture, ApertureRes, sin_alpha, obscuration);
    
    UpdateFADisplay();
    UpdatePlaneWaves();
}


void UpdateBADisplay() {
    tira::image<unsigned char> color = ColormapField(BackAperture, dispBackAperture);
    if (texBackAperture == NULL)
        texBackAperture = new tira::glTexture(color.data(), ApertureRes, ApertureRes, 0, GL_RGB, GL_RGB, GL_UNSIGNED_BYTE);
    else
        texBackAperture->AssignImage(color.data(), ApertureRes, ApertureRes, 0, GL_RGB, GL_RGB, GL_UNSIGNED_BYTE);

    texBackAperture->SetFilter(GL_NEAREST);
}
void UpdateBA() {
    BA_Gaussian(BackAperture, ApertureRes, polarization[0], polarization[1], beam_W0, gaussian_center[0], gaussian_center[1], lambda);
    
    UpdateBADisplay();
    UpdateFA();
}

void RenderRange(float field_width) {
    float half_width = field_width / 2;
    //output the field width-----------------
    std::stringstream stream_width;
    stream_width << std::fixed << std::setprecision(2) << half_width;
    std::string string_width = stream_width.str();
    ImGui::Columns(2);
    ImGui::Text("-%s", string_width.c_str());
    ImGui::NextColumn();
    ImGui::SetCursorPosX(ImGui::GetCursorPosX() + ImGui::GetColumnWidth() - ImGui::CalcTextSize(string_width.c_str()).x
        - ImGui::GetScrollX() - 2 * ImGui::GetStyle().ItemSpacing.x);
    ImGui::Text("%s", string_width.c_str());
    ImGui::NextColumn();
}


/// <summary>
/// This function renders the user interface every frame
/// </summary>
void RenderUI() {
    // Start the Dear ImGui frame
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    if (ImGui::GetIO().MouseClicked[1])
    {
        glfwGetCursorPos(window, &xpos, &ypos);
        ImGui::OpenPopup("save_slice");
    }
    if (ImGui::BeginPopup("save_slice"))
    {
        unsigned int N = pow(2, PSFExp);
        if (ImGui::Button("Save Slice")) {                                              // create a button that opens a file dialog
            ImGuiFileDialog::Instance()->OpenDialog("ChooseNpyFile", "Choose NPY File", ".npy,.npz", ".");
        }
        if (ImGuiFileDialog::Instance()->Display("ChooseNpyFile")) {				    // if the user opened a file dialog
            if (ImGuiFileDialog::Instance()->IsOk()) {								    // and clicks okay, they've probably selected a file
                std::string filename = ImGuiFileDialog::Instance()->GetFilePathName();	// get the name of the file
                std::string extension = filename.substr(filename.find_last_of(".") + 1);

                std::cout << "Cursor position: " << xpos << ", " << ypos << std::endl;
                std::cout << "File chosen: " << filename << std::endl;
                // RUIJIAO: determine which slice is clicked
                //          save the appropriate slice as an NPY file
                // Save the PSF
                if (true) {
                    const std::vector<long unsigned> shape{ (long unsigned)PSFRes, (long unsigned)PSFRes, 3 };
                    const bool fortran_order{ false };
                    npy::SaveArrayAsNumpy(filename, fortran_order, shape.size(), shape.data(), PSF.data());
                }

                // Wrong click at the upper left region
                else {
                    std::cout << "Wrong click at the wrong region. " << std::endl;
                    exit(1);
                }
            }
            ImGuiFileDialog::Instance()->Close();									// close the file dialog box
            ImGui::CloseCurrentPopup();
        }

        ImGui::EndPopup();
    }



    // Back Aperture Control Window
    {
        ImGui::Begin("Back Aperture");

        ImGui::Columns(2);
        ImGui::PushItemWidth(ImGui::GetColumnWidth() * item_width_scale);
        if (ImGui::InputInt("##ApertureExp", &ApertureExp, 1, 1)) {
            if (ApertureExp <= 0) ApertureExp = 1;
            ApertureRes = pow(2, ApertureExp) + 1;
            UpdateBA();
        }
        ImGui::NextColumn();
        ImGui::Text("(%d x %d)", ApertureRes, ApertureRes);

        ImGui::Columns(4);
        if (ImGui::RadioButton("X (real)", (int*)&dispBackAperture, XReal)) { UpdateBA(); }
        if (ImGui::RadioButton("X (imag)", (int*)&dispBackAperture, XImag)) { UpdateBA(); }
        ImGui::NextColumn();
        if (ImGui::RadioButton("Y (real)", (int*)&dispBackAperture, YReal)) { UpdateBA(); }
        if (ImGui::RadioButton("Y (imag)", (int*)&dispBackAperture, YImag)) { UpdateBA(); }
        ImGui::NextColumn();
        if (ImGui::RadioButton("Z (real)", (int*)&dispBackAperture, ZReal)) { UpdateBA(); }
        if (ImGui::RadioButton("Z (imag)", (int*)&dispBackAperture, ZImag)) { UpdateBA(); }
        ImGui::NextColumn();
        if (ImGui::RadioButton("I", (int*)&dispBackAperture, Intensity)) { UpdateBA(); }

        ImGui::Columns(1);  
        ImGui::Image((void*)texBackAperture->ID(), ImVec2(ImGui::GetColumnWidth(), ImGui::GetColumnWidth()));
        RenderRange(aperture_radius * 2);
        ImGui::Columns(2);
        ImGui::PushItemWidth(ImGui::GetColumnWidth() * item_width_scale);
        if (ImGui::InputFloat("##aperture_radius", &aperture_radius, 1, 100, "radius = %.1f")) {
            if (aperture_radius < 0) aperture_radius = 0;
            UpdateBA();
        }
        ImGui::NextColumn();
        ImGui::PushItemWidth(ImGui::GetColumnWidth() * item_width_scale);
        if (ImGui::DragFloat("##beam_width", &beam_W0, 1.0f, 0.0f, 3 * aperture_radius)) {
            UpdateBA();
        }
        ImGui::NextColumn();
        ImGui::PushItemWidth(ImGui::GetColumnWidth() * item_width_scale);
        if (ImGui::SliderFloat("##cx", &gaussian_center[0], -aperture_radius, aperture_radius, "cx = %.1f")) {
            UpdateBA();
        }
        ImGui::NextColumn();
        ImGui::PushItemWidth(ImGui::GetColumnWidth() * item_width_scale);
        if (ImGui::SliderFloat("##cy", &gaussian_center[1], -aperture_radius, aperture_radius, "cy = %.1f")) {
            UpdateBA();
        }

        

        ImGui::Columns(3);
        ImGui::PushItemWidth(ImGui::GetColumnWidth() * item_width_scale);
        if (ImGui::InputFloat("##min_lambda", &spectrum[0], 0.1f, 1.0f, "l0 = %.3f")) {
            if (spectrum[0] > spectrum[1]) spectrum[0] = spectrum[1];
            if (lambda < spectrum[0]) lambda = spectrum[0];
        }
        ImGui::NextColumn();
        ImGui::PushItemWidth(ImGui::GetColumnWidth() * item_width_scale);
        if (ImGui::SliderFloat("##lambda", &lambda, spectrum[0], spectrum[1], "lambda = %.3f")) {
            UpdateBA();
        }

        ImGui::NextColumn();
        ImGui::PushItemWidth(ImGui::GetColumnWidth() * item_width_scale);
        if (ImGui::InputFloat("##max_lambda", &spectrum[1], 0.1f, 1.0f, "l1 = %.3f")) {
            if (spectrum[1] < spectrum[0]) spectrum[1] = spectrum[0];
            if (lambda > spectrum[1]) lambda = spectrum[1];
        }
        
        
        ImGui::Columns(2);

        precision p0_real = polarization[0].real();
        precision p0_imag = polarization[0].imag();
        ImGui::PushItemWidth(ImGui::GetColumnWidth() * item_width_scale);
        if (ImGui::SliderFloat("##XPol_real", &p0_real, -1, 1)) {
            polarization[0] = std::complex<precision>(p0_real, p0_imag);
            UpdateBA();
        }
        if (ImGui::SliderFloat("##XPol_imag", &p0_imag, -1, 1)) {
            polarization[0] = std::complex<precision>(p0_real, p0_imag);
            UpdateBA();
        }
        ImGui::NextColumn();
        ImGui::PushItemWidth(ImGui::GetColumnWidth() * item_width_scale);
        precision p1_real = polarization[1].real();
        precision p1_imag = polarization[1].imag();
        if (ImGui::SliderFloat("##YPol_real", &p1_real, -1, 1)) {
            polarization[1] = std::complex<precision>(p1_real, p1_imag);
            UpdateBA();
        }
        if (ImGui::SliderFloat("##YPol_imag", &p1_imag, -1, 1)) {
            polarization[1] = std::complex<precision>(p1_real, p1_imag);
            UpdateBA();
        }
        ImGui::End();
    }

    {
        ImGui::Begin("Front Aperture");
        ImGui::Columns(2);
        ImGui::PushItemWidth(ImGui::GetColumnWidth() * item_width_scale);
        if (ImGui::InputInt("##ApertureExp", &ApertureExp, 1, 1)) {
            if (ApertureExp <= 0) ApertureExp = 1;
            ApertureRes = pow(2, ApertureExp) + 1;
            UpdateBA();
        }
        ImGui::NextColumn();
        ImGui::Text("(%d x %d)", ApertureRes, ApertureRes);

        ImGui::Columns(4);
        // Set the display method for the PSF field
        if (ImGui::RadioButton("X (real)", (int*)&dispFrontAperture, XReal)) { UpdateFA(); }
        if (ImGui::RadioButton("X (imag)", (int*)&dispFrontAperture, XImag)) { UpdateFA(); }
        ImGui::NextColumn();
        if (ImGui::RadioButton("Y (real)", (int*)&dispFrontAperture, YReal)) { UpdateFA(); }
        if (ImGui::RadioButton("Y (imag)", (int*)&dispFrontAperture, YImag)) { UpdateFA(); }
        ImGui::NextColumn();
        if (ImGui::RadioButton("Z (real)", (int*)&dispFrontAperture, ZReal)) { UpdateFA(); }
        if (ImGui::RadioButton("Z (imag)", (int*)&dispFrontAperture, ZImag)) { UpdateFA(); }
        ImGui::NextColumn();
        if (ImGui::RadioButton("I", (int*)&dispFrontAperture, Intensity)) { UpdateFA(); }
        

        ImGui::Columns(1);
        float display_width = ImGui::GetWindowWidth() * item_width_scale;
        ImGui::Image((void*)texFrontAperture->ID(), ImVec2(ImGui::GetColumnWidth(), ImGui::GetColumnWidth()));
        RenderRange(sin_alpha * 2);

        ImGui::Columns(3);
        ImGui::PushItemWidth(ImGui::GetColumnWidth() * item_width_scale);
        if (ImGui::SliderFloat("##sin_alpha", &sin_alpha, 0.0f, 1.0f, "sin(a) = %.2f")) {
            UpdateFA();
        }
        float NA = sin_alpha * refractive_index;
        ImGui::NextColumn();
        ImGui::Text("(%.3f NA)", NA);
        ImGui::NextColumn();
        ImGui::PushItemWidth(ImGui::GetColumnWidth() * item_width_scale);
        float obs_percent = obscuration * 100;
        if (ImGui::SliderFloat("##obscuration", &obs_percent, 0.0f, 100.0f, "obs = %.0f%%")) {
            obscuration = obs_percent / 100.0f;
            UpdateFA();
        }
        
        ImGui::End();
    }

    {
        ImGui::Begin("Point Spread Function");
        ImGui::Columns(2);
        ImGui::PushItemWidth(ImGui::GetColumnWidth()* item_width_scale * 0.5);
        if (ImGui::InputInt("Resolution", &PSFExp, 1, 1)) {
            if (PSFExp <= 0) PSFExp = 1;
            PSFRes = pow(2, PSFExp) + 1;
            UpdatePSF();
        }
        if (ImGui::InputInt("CUDA Device", &device, 1, 1)) {
            if (device < -1) device = -1;
            if (device >= devices - 1) device = devices - 1;
            UpdatePSF();
        }
        
        ImGui::NextColumn();
        ImGui::Text("(%d x %d)", PSFRes, PSFRes);

        ImGui::Columns(4);
        // Set the display method for the PSF field
        if (ImGui::RadioButton("X (real)", (int*)&dispPSF, XReal)) { UpdatePSFDisplay(); }
        if (ImGui::RadioButton("X (imag)", (int*)&dispPSF, XImag)) { UpdatePSFDisplay(); }
        ImGui::NextColumn();
        if (ImGui::RadioButton("Y (real)", (int*)&dispPSF, YReal)) { UpdatePSFDisplay(); }
        if (ImGui::RadioButton("Y (imag)", (int*)&dispPSF, YImag)) { UpdatePSFDisplay(); }
        ImGui::NextColumn();
        if (ImGui::RadioButton("Z (real)", (int*)&dispPSF, ZReal)) { UpdatePSFDisplay(); }
        if (ImGui::RadioButton("Z (imag)", (int*)&dispPSF, ZImag)) { UpdatePSFDisplay(); }
        ImGui::NextColumn();
        if (ImGui::RadioButton("I", (int*)&dispPSF, Intensity)) { UpdatePSFDisplay(); }
        if (ImGui::RadioButton("SUM(I)", (int*)&dispPSF, IncoherentIntensity)) { 
            UpdateIncoherent();
        }

        

        ImGui::Columns(1);
        ImGui::Image((void*)texPSF->ID(), ImVec2(ImGui::GetColumnWidth(), ImGui::GetColumnWidth()));
        ImGui::SetItemUsingMouseWheel();
        if (ImGui::IsItemHovered()) {
            float mousewheel_scale = 0.1;
            if (axis == 0) planes[0] += ImGui::GetIO().MouseWheel * mousewheel_scale;
            if (axis == 1) planes[1] += ImGui::GetIO().MouseWheel * mousewheel_scale;
            if (axis == 2) planes[2] += ImGui::GetIO().MouseWheel * mousewheel_scale;

            if (ImGui::GetIO().MouseWheel != 0) UpdatePSF();
        }
        RenderRange(psf_window_width);
        
        
        ImGui::Columns(2);
        //ImGui::NextColumn();
        if (ImGui::RadioButton("X", &axis, 0)) { UpdatePSF(); }
        ImGui::SameLine();
        if (ImGui::RadioButton("Y", &axis, 1)) { UpdatePSF(); }
        ImGui::SameLine();
        if (ImGui::RadioButton("Z", &axis, 2)) { UpdatePSF(); }
        ImGui::NextColumn();
        ImGui::PushItemWidth(ImGui::GetColumnWidth()* item_width_scale);
        if (axis == 0) {
            if (ImGui::InputFloat("##x_plane", &planes[0], 0.1f, 1.0f, "x = %.1f")) {
                UpdatePSF();
            }
        }
        else if (axis == 1) {
            if (ImGui::InputFloat("##y_plane", &planes[1], 0.1f, 1.0f, "y = %.1f")) {
                UpdatePSF();
            }
        }
        else {
            if (ImGui::InputFloat("##z_plane", &planes[2], 0.1f, 1.0f, "z = %.1f")) {
                UpdatePSF();
            }
        }        

        ImGui::Columns(2);
        ImGui::NextColumn();
        ImGui::PushItemWidth(ImGui::GetColumnWidth()* item_width_scale);
        if (ImGui::InputFloat("##width", &psf_window_width, 1.0f, 10.0f, "width = %.2f")) {
            if (psf_window_width < 1.0f) psf_window_width = 1.0f;
            psf_extent = psf_window_width / 2.0f;
            UpdatePSF();
        }
        ImGui::NextColumn();
        ImGui::PushItemWidth(ImGui::GetColumnWidth() * item_width_scale);
        if (ImGui::InputFloat("##RI", &refractive_index, 0.1, 0.5, "ni = %.2f")) {
            UpdatePlaneWaves();
        }
        ImGui::NextColumn();
        if (ImGui::Checkbox("substrate", &substrate)) {
            UpdatePlaneWaves();
        }
        if (ImGui::Checkbox("Incident", &PSFIncident)) {
            UpdatePSF();
        }
        if (ImGui::Checkbox("Reflected", &PSFReflected)) {
            UpdatePSF();
        }
        if (ImGui::Checkbox("Transmitted", &PSFTransmitted)) {
            UpdatePSF();
        }
        ImGui::NextColumn();
        ImGui::PushItemWidth(ImGui::GetColumnWidth()* item_width_scale);
        if (ImGui::InputFloat("##substrate_n", &substrate_n_real, 0.1, 0.5, "nt = %.2f")) {
            UpdatePlaneWaves();
        }
        if (ImGui::InputFloat("##substrate_k", &substrate_n_imag, 0.01, 0.05, "kappa = %.2f")) {
            UpdatePlaneWaves();
        }


        
        ImGui::End();
    }
    

    
    //ImGui::Text("PSF Calculation: %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);  // Render a separate window showing the FPS
    ImGui::Text("PSF Calculation: %.3f ms", psf_time * 1000);  // Render a separate window showing the FPS
    if(dispPSF == IncoherentIntensity)
        ImGui::Text("Integration frames for incoherent imaging: %d", IncoherentCounter);
    ImGui::Render();                                                            // Render all windows
}

/// <summary>
/// Initialize the GUI
/// </summary>
/// <param name="window">Pointer to the GLFW window that will be used for rendering</param>
/// <param name="glsl_version">Version of GLSL that will be used</param>
void InitUI(GLFWwindow* window, const char* glsl_version) {
    // Setup Dear ImGui context
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;

    // Setup Dear ImGui style
    ImGui::StyleColorsDark();
    ImGui::GetStyle().ScaleAllSizes(ui_scale);
    ImGui::GetIO().FontGlobalScale = ui_scale;

    // Setup Platform/Renderer backends
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init(glsl_version);

}

/// <summary>
/// Destroys the ImGui rendering interface (usually called when the program closes)
/// </summary>
void DestroyUI() {
    // Cleanup
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
}
static void glfw_error_callback(int error, const char* description) {
    fprintf(stderr, "Glfw Error %d: %s\n", error, description);
}

int main(int argc, char** argv) {
    // Setup window
    glfwSetErrorCallback(glfw_error_callback);
    if (!glfwInit())
        return 1;

    if (!glewInit()) {
        return 1;
    }

    // GL 3.0 + GLSL 130
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);

    // Create window with graphics context
    window = glfwCreateWindow(1600, 1200, "ImGui GLFW+OpenGL3 Hello World Program", NULL, NULL);
    if (window == NULL)
        return 1;
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1); // Enable vsync

    InitUI(window, glsl_version);

    cudaGetDeviceCount(&devices);
    if (device > devices - 1)
        device = devices - 1;

    // Initialize the random number generator;
    rng.seed(time(NULL));
    //lambda_distribution = std::uniform_real_distribution<float>(spectrum[0], spectrum[1]);

    /// Initialize apertures
    UpdateBA();

    

    

    // Main loop
    while (!glfwWindowShouldClose(window)) {

        if (dispPSF == IncoherentIntensity) {
            UpdateIncoherent();
        }
        // Poll and handle events (inputs, window resize, etc.)
        glfwPollEvents();

        RenderUI();
        int display_w, display_h;
        glfwGetFramebufferSize(window, &display_w, &display_h);

        glViewport(0, 0, display_w, display_h);                     // specifies the area of the window where OpenGL can render
        glClearColor(clear_color.x * clear_color.w, clear_color.y * clear_color.w, clear_color.z * clear_color.w, clear_color.w);

        glClear(GL_COLOR_BUFFER_BIT);                               // clear the Viewport using the clear color

        /****************************************************/
        /*      Draw Stuff To The Viewport                  */
        /****************************************************/



        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());     // draw the GUI data from its buffer

        glfwSwapBuffers(window);                                    // swap the double buffer
    }

    
    DestroyUI();                                                    // Clear the ImGui user interface

    glfwDestroyWindow(window);                                      // Destroy the GLFW rendering window
    glfwTerminate();                                                // Terminate GLFW

    return 0;
}