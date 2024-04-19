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

#include <iostream>
#include <string>
#include <stdio.h>
#include <numbers>

#include <cuda/std/complex>
#include "cuda_runtime.h"

void gpuPSF(cuda::std::complex<float>* field, unsigned int field_res, float extent, float w, int axis,
    cuda::std::complex<float>* aperture, float sin_alpha, unsigned int fa_res, float refractive_index, float lambda, int device);

void gpuPSFSubstrate(cuda::std::complex<float>* field, unsigned int field_res, float extent, float w, int axis, cuda::std::complex<float> nr,
    cuda::std::complex<float>* aperture, float sin_alpha, unsigned int fa_res, float refractive_index, float lambda, int device);

// User interface variables
GLFWwindow* window;                                     // pointer to the GLFW window that will be created (used in GLFW calls to request properties)
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
enum FieldDisplay {XReal, XImag, YReal, YImag, ZReal, ZImag, Intensity, Magnitude};
int ApertureExp = 6;                                    // exponent for the back aperture resolution (always a power of 2)
int ApertureRes = pow(2, ApertureExp) + 1;
int PSFExp = 6;                                         // exponent for the PSF resolution
int PSFRes = pow(2, PSFExp) + 1;
tira::image<std::complex<precision>> BackAperture;      // define the back aperture image
tira::glTexture* texBackAperture;
FieldDisplay dispBackAperture = XReal;

tira::image<std::complex<precision>> FrontAperture;     // define an image array for the front aperture
tira::glTexture* texFrontAperture;
FieldDisplay dispFrontAperture = XReal;

tira::image<std::complex<precision>> PSF;               // image to display the point spread function
tira::glTexture* texPSF;
FieldDisplay dispPSF = XReal;

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

// UI Elements
precision display_size = 500;                             // size of each field image on the screen

bool substrate = true;
precision substrate_n_real = 1.4f;
precision substrate_n_imag = 0.0f;



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

void BA_Gaussian(unsigned int resolution, std::complex<precision> xp, std::complex<precision> yp, precision width, precision amplitude = 1, precision lambda = 1, precision z = 0) {

    unsigned int N = resolution;
    BackAperture.resize({ N, N, 3 });
    precision x, y, y_sq;
    precision d = (2 * aperture_radius) / (N - 1);

    precision k = 2 * std::numbers::pi / lambda;
    precision z0 = std::numbers::pi / lambda * pow(width, 2);

    for (unsigned int yi = 0; yi < N; yi++) {
        y = -aperture_radius + d * yi;
        y_sq = y * y;
        for (unsigned int xi = 0; xi < N; xi++) {
            x = -aperture_radius + d * xi;
            precision rho_sq = y_sq + x * x;
            std::complex<precision> A0 = amplitude / std::complex<precision>(0, z0);
            precision zeta = atan(z / z0);

            precision Wz = width * sqrt(1 + pow(z / z0, 2));
            precision U1 = amplitude * (width / Wz);
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
            BackAperture(xi, yi, 0) = xp * U;
            BackAperture(xi, yi, 1) = yp * U;
            BackAperture(xi, yi, 2) = 0;
        }
    }
}

// Optical functions that calculate the field at the front aperture of the objective
void FA(unsigned int resolution, precision max_sin_alpha, precision obscuration) {

    unsigned int N = resolution;
    precision sin_alpha = max_sin_alpha;
    precision dsa = (2 * sin_alpha) / (N - 1);

    FrontAperture.resize({ N, N, 3 });
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
                FA[0] = sqrt_sz_inv * (C[0][0] * BackAperture(xi, yi, 0) + C[0][1] * BackAperture(xi, yi, 1) + C[0][2] * BackAperture(xi, yi, 2));
                FA[1] = sqrt_sz_inv * (C[1][0] * BackAperture(xi, yi, 0) + C[1][1] * BackAperture(xi, yi, 1) + C[1][2] * BackAperture(xi, yi, 2));
                FA[2] = sqrt_sz_inv * (C[2][0] * BackAperture(xi, yi, 0) + C[2][1] * BackAperture(xi, yi, 1) + C[2][2] * BackAperture(xi, yi, 2));
                FrontAperture(xi, yi, 0) = FA[0];
                FrontAperture(xi, yi, 1) = FA[1];
                FrontAperture(xi, yi, 2) = FA[2];
            }
            else {
                FrontAperture(xi, yi, 0) = 0;
                FrontAperture(xi, yi, 1) = 0;
                FrontAperture(xi, yi, 2) = 0;
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

void cpuPSFSubstrate(complex<float>* field, unsigned int field_res, float extent, float w, int axis, complex<float> nr,
    complex<float>* aperture, float sin_alpha, unsigned int fa_res, float refractive_index, float lambda) {
    unsigned int N = field_res;
    float d = (2 * extent) / (N - 1);

    float ds = (2 * sin_alpha) / (fa_res - 1);

    float k_mag = 2 * std::numbers::pi / lambda * refractive_index;
    tira::vec3<float> s;
    float sin_alpha_2 = sin_alpha * sin_alpha;

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
                    if (sx2_sy2 <= sin_alpha_2) {

                        s[2] = sqrt(1 - sx2_sy2);
                        size_t i = syi * fa_res * 3 + sxi * 3;
                        tira::cvec3<float> A(aperture[i + 0], aperture[i + 1], aperture[i + 2]);
                        tira::vec3<float> k(k_mag * s[0], k_mag * s[1], k_mag * s[2]);
                        float k_dot_r;
                        tira::planewave<float> p(k[0], k[1], k[2], A[0], A[1], A[2]);
                        tira::planewave<float> r, t;
                        
                        if (axis == 0) {
                            if (v <= 0) {
                                E += p.E(w, u, v);                               
                                E += p.reflect(nr).E(w, u, v);
                            }
                            else {                                
                                E += p.refract(nr).E(w, u, v);
                            }
                        }
                        else if (axis == 1) {
                            if (v <= 0) {
                                E += p.E(u, w, v);
                                E += p.reflect(nr).E(u, w, v);
                            }
                            else {
                                E += p.refract(nr).E(u, w, v);
                            }
                        }
                        else {
                            if (w <= 0) {
                                E += p.E(u, v, w);
                                E += p.reflect(nr).E(u, v, w);
                            }
                            else {
                                E += p.refract(nr).E(u, w, v);
                            }
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

    case Magnitude:
        scalar = getFieldMagnitude(E);
        maxabs = scalar.maxv();
        return scalar.cmap(0, maxabs, cmapIntensity);
    }

    maxabs = std::max(abs(scalar.minv()), abs(scalar.maxv()));
    return scalar.cmap(-maxabs, maxabs, cmapDiverging);
}

void UpdatePSFDisplay() {
    tira::image<unsigned char> color = ColormapField(PSF, dispPSF);
    texPSF = new tira::glTexture(color.data(), PSFRes, PSFRes, 0, GL_RGB, GL_RGB, GL_UNSIGNED_BYTE);
    texPSF->SetFilter(GL_NEAREST);
}
void UpdatePSF() {
    
    PSF.resize({ (size_t)PSFRes, (size_t)PSFRes, 3 });

    if (substrate) {
        if (device < 0) {
            cpuPSFSubstrate(PSF.data(), PSFRes, psf_extent, planes[axis], axis, refractive_index / complex<float>(substrate_n_real, substrate_n_imag),
                FrontAperture.data(), sin_alpha, ApertureRes, refractive_index, lambda);
        }
        else {
            gpuPSFSubstrate((cuda::std::complex<float>*)PSF.data(), PSFRes, psf_extent, planes[axis], axis, 
                cuda::std::complex<float>(substrate_n_real, substrate_n_imag) / cuda::std::complex<float>(refractive_index, 0),
                (cuda::std::complex<float>*)FrontAperture.data(), sin_alpha, ApertureRes, refractive_index, lambda, device);
        }
    }
    else {
        if (device < 0)
            cpuPSF(PSF.data(), PSFRes, psf_extent, planes[axis], axis, FrontAperture.data(), sin_alpha, ApertureRes, refractive_index, lambda);
        else
            gpuPSF((cuda::std::complex<float>*)PSF.data(), PSFRes, psf_extent, planes[axis], axis, (cuda::std::complex<float>*)FrontAperture.data(), sin_alpha, ApertureRes, refractive_index, lambda, device);
    }
    
    UpdatePSFDisplay();
}

void UpdateFADisplay() {
    tira::image<unsigned char> color = ColormapField(FrontAperture, dispFrontAperture);
    texFrontAperture = new tira::glTexture(color.data(), ApertureRes, ApertureRes, 0, GL_RGB, GL_RGB, GL_UNSIGNED_BYTE);
    texFrontAperture->SetFilter(GL_NEAREST);
}
void UpdateFA() {
    FA(ApertureRes, sin_alpha, obscuration);
    
    UpdateFADisplay();
    UpdatePSF();
}


void UpdateBADisplay() {
    tira::image<unsigned char> color = ColormapField(BackAperture, dispBackAperture);
    texBackAperture = new tira::glTexture(color.data(), ApertureRes, ApertureRes, 0, GL_RGB, GL_RGB, GL_UNSIGNED_BYTE);
    texBackAperture->SetFilter(GL_NEAREST);
}
void UpdateBA() {
    BA_Gaussian(ApertureRes, polarization[0], polarization[1], beam_W0);
    
    UpdateBADisplay();
    UpdateFA();
}

void RenderTexture(GLuint texID, float window_width, float field_width, float colorbar_min, float colorbar_max) {
    ImGui::Image((void*)texID, ImVec2(window_width, window_width));

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
    ImGui::Columns(1);
    //------------------------------------------------
}


/// <summary>
/// This function renders the user interface every frame
/// </summary>
void RenderUI() {
    // Start the Dear ImGui frame
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    // Back Aperture Control Window
    {
        ImGui::Begin("Back Aperture");

        ImGui::Columns(2);
        if (ImGui::InputInt("##ApertureExp", &ApertureExp, 1, 1)) {
            if (ApertureExp <= 0) ApertureExp = 1;
            ApertureRes = pow(2, ApertureExp) + 1;
            UpdateBA();
        }
        ImGui::NextColumn();
        ImGui::Text("(%d x %d)", ApertureRes, ApertureRes);
        ImGui::Columns(1);

        float display_width = ImGui::GetWindowWidth() * item_width_scale;
        RenderTexture(texBackAperture->ID(), display_width, aperture_radius * 2, 0, 1);
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
        if (ImGui::RadioButton("|E|", (int*)&dispBackAperture, Magnitude)) { UpdateBA(); }

        ImGui::Columns(3);
        if (ImGui::InputFloat("##min_lambda", &spectrum[0], 0.1f, 1.0f, "l0 = %.3f")) {
            if (lambda < spectrum[0]) lambda = spectrum[0];
        }
        ImGui::NextColumn();
        if (ImGui::SliderFloat("##lambda", &lambda, spectrum[0], spectrum[1], "lambda = %.3f")) {
            UpdateBA();
        }

        ImGui::NextColumn();
        if (ImGui::InputFloat("##max_lambda", &spectrum[1], 0.1f, 1.0f, "l1 = %.3f")) {
            if (lambda > spectrum[1]) lambda = spectrum[1];
        }
        
        
        ImGui::Columns(1);

        ImGui::PushStyleVar(ImGuiStyleVar_GrabMinSize, 40);
        precision p0_real = polarization[0].real();
        precision p0_imag = polarization[0].imag();
        if (ImGui::VSliderFloat("##XPol_real", ImVec2(100, 160), &p0_real, -1, 1)) {
            polarization[0] = std::complex<precision>(p0_real, p0_imag);
            UpdateBA();
        }
        ImGui::SameLine();
        if (ImGui::VSliderFloat("##XPol_imag", ImVec2(100, 160), &p0_imag, -1, 1)) {
            polarization[0] = std::complex<precision>(p0_real, p0_imag);
            UpdateBA();
        }
        
        ImGui::SameLine();
        precision p1_real = polarization[1].real();
        precision p1_imag = polarization[1].imag();
        if (ImGui::VSliderFloat("##YPol_real", ImVec2(100, 160), &p1_real, -1, 1)) {
            polarization[1] = std::complex<precision>(p1_real, p1_imag);
            UpdateBA();
        }
        ImGui::SameLine();
        if (ImGui::VSliderFloat("##YPol_imag", ImVec2(100, 160), &p1_imag, -1, 1)) {
            polarization[1] = std::complex<precision>(p1_real, p1_imag);
            UpdateBA();
        }
        ImGui::PopStyleVar();
        ImGui::End();
    }

    {
        ImGui::Begin("Front Aperture");
        ImGui::Columns(2);
        if (ImGui::InputInt("##ApertureExp", &ApertureExp, 1, 1)) {
            if (ApertureExp <= 0) ApertureExp = 1;
            ApertureRes = pow(2, ApertureExp) + 1;
            UpdateBA();
        }
        ImGui::NextColumn();
        ImGui::Text("(%d x %d)", ApertureRes, ApertureRes);
        ImGui::Columns(1);
        float display_width = ImGui::GetWindowWidth() * item_width_scale;
        RenderTexture(texFrontAperture->ID(), display_width, sin_alpha * 2, 0, 1);

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

        ImGui::Columns(4);

        if (ImGui::RadioButton("X (real)", (int*)&dispFrontAperture, XReal)) { UpdateBA(); }
        if (ImGui::RadioButton("X (imag)", (int*)&dispFrontAperture, XImag)) { UpdateBA(); }
        ImGui::NextColumn();        
        if (ImGui::RadioButton("Y (real)", (int*)&dispFrontAperture, YReal)) { UpdateBA(); }
        if (ImGui::RadioButton("Y (imag)", (int*)&dispFrontAperture, YImag)) { UpdateBA(); }
        ImGui::NextColumn();        
        if (ImGui::RadioButton("Z (real)", (int*)&dispFrontAperture, ZReal)) { UpdateBA(); }
        if (ImGui::RadioButton("Z (imag)", (int*)&dispFrontAperture, ZImag)) { UpdateBA(); }
        ImGui::NextColumn();
        if (ImGui::RadioButton("I", (int*)&dispFrontAperture, Intensity)) { UpdateBA(); }
        if (ImGui::RadioButton("|E|", (int*)&dispFrontAperture, Magnitude)) { UpdateBA(); }
        
        ImGui::End();
    }

    {
        ImGui::Begin("Point Spread Function");
        ImGui::Columns(2);
        
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
        ImGui::Columns(1);
        
        float display_width = ImGui::GetWindowWidth() * item_width_scale;
        RenderTexture(texPSF->ID(), display_width, psf_window_width, 0, 1);
        
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
        if (ImGui::RadioButton("|E|", (int*)&dispPSF, Magnitude)) { UpdatePSFDisplay(); }

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
        if (ImGui::InputFloat("##RI", &refractive_index, 0.1, 0.5, "n0 = %.2f")) {
            UpdatePSF();
        }
        if (ImGui::Checkbox("substrate", &substrate)) {
            UpdatePSF();
        }
        ImGui::NextColumn();
        if (ImGui::InputFloat("##substrate_n", &substrate_n_real, 0.1, 0.5, "n = %.2f")) {
            UpdatePSF();
        }
        if (ImGui::InputFloat("##substrate_k", &substrate_n_imag, 0.01, 0.05, "kappa = %.2f")) {
            UpdatePSF();
        }


        
        ImGui::End();
    }
    

    
    ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);  // Render a separate window showing the FPS

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

    /// Initialize apertures
    UpdateBA();

    cudaGetDeviceCount(&devices);
    if (device > devices - 1)
        device = devices - 1;

    // Main loop
    while (!glfwWindowShouldClose(window)) {
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