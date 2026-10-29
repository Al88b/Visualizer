
//g++ vumetres.cxx -o live_visualizer -lglfw -lGL -lm -lpthread -ldl
// Run from the repo folder so the TV frame PNG is found, or pass its path as the first argument.

#include <GLFW/glfw3.h>
#include <cmath>
#include <vector>
#include <iostream>
#include <complex>
#include <atomic>
#include <string>

// Define miniaudio implementation ONLY in one source file
#define MA_NO_JACK
#define MINIAUDIO_IMPLEMENTATION
#include "miniaudio.h"

// PNG loading for the TV frame overlay
#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_PNG
#include "stb_image.h"

// Constants
// Window matches the TV frame image's aspect ratio (800x645) so it isn't distorted
const int WINDOW_WIDTH = 800;
const int WINDOW_HEIGHT = 645;
const int BUFFER_SIZE = 512; // Must be a power of 2 for the FFT
const int EQ_BANDS = 32;

// ---------------- TV FRAME OVERLAY ----------------
const char* TV_FRAME_PATH = "istockphoto-157505991-612x612.png";

// The transparent "screen" hole inside the TV image, as fractions of the image
// size (image is 800x645, hole spans x 160..653, y 78..470 from the top).
// The visualizer is rendered into this rectangle and the frame is drawn on top,
// so the rounded corners of the bezel neatly mask the edges.
const float SCREEN_LEFT   = 160.0f / 800.0f;
const float SCREEN_RIGHT  = 654.0f / 800.0f;
const float SCREEN_TOP    =  78.0f / 645.0f;
const float SCREEN_BOTTOM = 471.0f / 645.0f;

GLuint g_tvTexture = 0;

// Aspect correction (height / width) of the area the visualizer is drawn into.
// Updated every frame so the VU meter arcs stay circular.
float g_viewAspect = (float)WINDOW_HEIGHT / WINDOW_WIDTH;

// Shared safe buffers between Audio thread and OpenGL thread
std::vector<float> g_waveformBuffer(BUFFER_SIZE, 0.0f);
std::vector<float> g_fftBands(EQ_BANDS, 0.0f);

// VU meters: per-channel RMS written by the audio thread, read by the render thread.
// std::atomic<float> keeps the cross-thread read/write safe without a mutex.
std::atomic<float> g_rmsLeft{0.0f};
std::atomic<float> g_rmsRight{0.0f};

// Needle positions (0 = rest, 1 = full scale). Only touched on the render thread.
float g_vuNeedleLeft  = 0.0f;
float g_vuNeedleRight = 0.0f;

// Helper function to map a Hue value [0.0, 1.0] to RGB components
void getRainbowColor(float hue, float& r, float& g, float& b) {
    float h = hue * 6.0f;
    int i = (int)floor(h);
    float f = h - i;
    float q = 1.0f - f;

    switch (i % 6) {
        case 0: r = 1.0f; g = f;    b = 0.0f; break; // Red to Yellow
        case 1: r = q;    g = 1.0f; b = 0.0f; break; // Yellow to Green
        case 2: r = 0.0f; g = 1.0f; b = f;    break; // Green to Cyan
        case 3: r = 0.0f; g = q;    b = 1.0f; break; // Cyan to Blue
        case 4: r = f;    g = 0.0f; b = 1.0f; break; // Blue to Magenta
        case 5: r = 1.0f; g = 0.0f; b = q;    break; // Magenta to Red
    }
}

// Simple Cooley-Tukey FFT algorithm implementation
void fft(std::vector<std::complex<float>>& x) {
    size_t N = x.size();
    if (N <= 1) return;

    std::vector<std::complex<float>> even(N / 2), odd(N / 2);
    for (size_t i = 0; i < N / 2; ++i) {
        even[i] = x[2 * i];
        odd[i] = x[2 * i + 1];
    }

    fft(even);
    fft(odd);

    for (size_t k = 0; k < N / 2; ++k) {
        std::complex<float> t = std::polar(1.0f, static_cast<float>(-2.0f * M_PI * k / N)) * odd[k];

        x[k] = even[k] + t;
        x[k + N / 2] = even[k] - t;
    }
}

// Audio Capture Callback (Called automatically by miniaudio background thread)
void audioCallback(ma_device* pDevice, void* pOutput, const void* pInput, ma_uint32 frameCount) {
    if (pInput == nullptr) return;

    const float* fInput = (const float*)pInput;
    unsigned int channels = pDevice->capture.channels;

    // 0. Per-channel RMS for the VU meters (before we mix down to mono)
    if (frameCount > 0) {
        float sumL = 0.0f, sumR = 0.0f;
        for (ma_uint32 i = 0; i < frameCount; ++i) {
            float l = fInput[i * channels];
            float r = (channels > 1) ? fInput[i * channels + 1] : l;
            sumL += l * l;
            sumR += r * r;
        }
        g_rmsLeft.store(std::sqrt(sumL / frameCount));
        g_rmsRight.store(std::sqrt(sumR / frameCount));
    }

    // 1. Process Time-Domain Data (Oscilloscope)
    // Convert multi-channel input to mono by averaging channels
    std::vector<float> localWaveform(BUFFER_SIZE, 0.0f);
    for (unsigned int i = 0; i < frameCount && i < BUFFER_SIZE; ++i) {
        float monoSample = 0.0f;
        for (unsigned int c = 0; c < channels; ++c) {
            monoSample += fInput[i * channels + c];
        }
        localWaveform[i] = monoSample / channels;
    }

    // 2. Process Frequency-Domain Data (FFT for EQ)
    std::vector<std::complex<float>> fftBuffer(BUFFER_SIZE);
    for (int i = 0; i < BUFFER_SIZE; ++i) {
        // Apply a Hanning Window to smooth out the audio chunk boundaries
        float window = 0.5f * (1.0f - std::cos(2.0f * M_PI * i / (BUFFER_SIZE - 1)));
        fftBuffer[i] = std::complex<float>(localWaveform[i] * window, 0.0f);
    }

    fft(fftBuffer);

    // Group raw frequencies into log-scaled EQ bands
    std::vector<float> localBands(EQ_BANDS, 0.0f);
    int samplesPerBand = (BUFFER_SIZE / 2) / EQ_BANDS; 
    for (int i = 0; i < EQ_BANDS; ++i) {
        float magnitudeSum = 0.0f;
        for (int j = 0; j < samplesPerBand; ++j) {
            int idx = i * samplesPerBand + j;
            magnitudeSum += std::abs(fftBuffer[idx]);
        }
        // Normalize and apply multiplier for visual pop
        localBands[i] = (magnitudeSum / samplesPerBand) * 4.0f; 
        if (localBands[i] > 1.0f) localBands[i] = 1.0f; // Clamp
    }

    // Thread-safe update to global containers
    g_waveformBuffer = localWaveform;
    g_fftBands = localBands;

    (void)pOutput; // Unused in capture-only mode
}

void drawOscilloscope() {
    glLineWidth(3.0f); // Thicker line makes the rainbow stand out better

    glBegin(GL_LINE_STRIP);
    for (int i = 0; i < BUFFER_SIZE; ++i) {
        float x = -1.0f + 2.0f * ((float)i / (BUFFER_SIZE - 1));
        float rawSample = g_waveformBuffer[i] * 0.45f;
        float y = rawSample + 0.5f;

        // Rainbow Left-to-Right: Hue relies directly on structural array position
        float hue = (float)i / (BUFFER_SIZE - 1);
        float r, g, b;
        getRainbowColor(hue, r, g, b);
        
        glColor3f(r, g, b); // Dynamically set color for this point on the wave
        glVertex2f(x, y);
    }
    glEnd();

    // Subtle dividing center line
    glColor3f(0.12f, 0.12f, 0.15f);
    glLineWidth(1.0f);
    glBegin(GL_LINES);
        glVertex2f(-1.0f, 0.0f);
        glVertex2f(1.0f, 0.0f);
    glEnd();
}


void drawEQGraph() {
    float spacing = 0.01f;
    float totalWidth = 2.0f;
    float barWidth = (totalWidth - (spacing * (EQ_BANDS + 1))) / EQ_BANDS;

    for (int i = 0; i < EQ_BANDS; ++i) {
        float xLeft = -1.0f + spacing + i * (barWidth + spacing);
        float xRight = xLeft + barWidth;
        float yBottom = -1.0f; 
        float yTop = yBottom + (g_fftBands[i] * 1.0f); 

        // Rainbow Right-to-Left: Invert the lookup index calculation
        float hue = (float)(EQ_BANDS - 1 - i) / (EQ_BANDS - 1);
        float r, g, b;
        getRainbowColor(hue, r, g, b);

        glBegin(GL_QUADS);
            // Bottom of the bar: slightly darker/shaded version for depth
            glColor3f(r * 0.4f, g * 0.4f, b * 0.4f); 
            glVertex2f(xLeft, yBottom);
            glVertex2f(xRight, yBottom);
            
            // Top of the bar: Full vibrant rainbow intensity
            glColor3f(r, g, b); 
            glVertex2f(xRight, yTop);
            glVertex2f(xLeft, yTop);
        glEnd();
    }
}

// ---------------- VU METERS ----------------

// Scale runs from -20 VU to +3 VU. 0 VU is aligned to -18 dBFS (the usual
// digital reference), so normal music sits around the middle of the dial.
const float VU_MIN = -20.0f;
const float VU_MAX = 3.0f;
const float VU_REF_DBFS = -9.0f;
const float VU_SWEEP = 0.8f;   // half-sweep of the needle in radians (~46 degrees)

// Classic VU dials are laid out linearly in *amplitude*, not dB, which is why
// the low end is squashed and 0 VU sits about two-thirds of the way across.
float vuToDial(float vu) {
    float a    = std::pow(10.0f, vu / 20.0f);
    float aMin = std::pow(10.0f, VU_MIN / 20.0f);
    float aMax = std::pow(10.0f, VU_MAX / 20.0f);
    float t = (a - aMin) / (aMax - aMin);
    return t < 0.0f ? 0.0f : (t > 1.0f ? 1.0f : t);
}

float rmsToDial(float rms) {
    float dbfs = 20.0f * std::log10(rms + 1e-9f);
    return vuToDial(dbfs - VU_REF_DBFS);
}

// Needle ballistics: quick-ish rise, slower fall, frame-rate independent.
void updateVUMeters(float dt) {
    auto step = [dt](float& needle, float target) {
        float tau = (target > needle) ? 0.06f : 0.18f;   // seconds
        needle += (target - needle) * (1.0f - std::exp(-dt / tau));
    };
    step(g_vuNeedleLeft,  rmsToDial(g_rmsLeft.load()));
    step(g_vuNeedleRight, rmsToDial(g_rmsRight.load()));
}

// Draws one analog-style meter. (cx, bottom) is the centre-bottom of the face,
// sizes are in normalised device coords. needle is 0..1 across the dial.
void drawVUMeter(float cx, float bottom, float faceW, float faceH, float needle) {
    // The window isn't square, so squash x to keep the arc circular
    const float xs = g_viewAspect;
    const float pivotY = bottom + faceH * 0.08f;
    const float R = faceH * 0.80f;
    const float zeroT = vuToDial(0.0f);

    auto dialPoint = [&](float t, float radius, float& x, float& y) {
        float ang = -VU_SWEEP + t * 2.0f * VU_SWEEP;   // 0 = straight up
        x = cx + radius * std::sin(ang) * xs;
        y = pivotY + radius * std::cos(ang);
    };

    float left = cx - faceW * 0.5f, right = cx + faceW * 0.5f, top = bottom + faceH;

    // Face: translucent so the bars still show through underneath
    glColor4f(0.05f, 0.05f, 0.07f, 0.78f);
    glBegin(GL_QUADS);
        glVertex2f(left, bottom);  glVertex2f(right, bottom);
        glVertex2f(right, top);    glVertex2f(left, top);
    glEnd();

    // Bezel
    glLineWidth(2.0f);
    glColor4f(0.35f, 0.35f, 0.42f, 1.0f);
    glBegin(GL_LINE_LOOP);
        glVertex2f(left, bottom);  glVertex2f(right, bottom);
        glVertex2f(right, top);    glVertex2f(left, top);
    glEnd();

    // Scale arc: grey up to 0 VU, red above it
    glLineWidth(3.0f);
    const int SEGS = 48;
    for (int pass = 0; pass < 2; ++pass) {
        float t0 = pass == 0 ? 0.0f : zeroT;
        float t1 = pass == 0 ? zeroT : 1.0f;
        if (pass == 0) glColor4f(0.75f, 0.75f, 0.8f, 1.0f);
        else           glColor4f(1.0f, 0.15f, 0.15f, 1.0f);
        glBegin(GL_LINE_STRIP);
        for (int i = 0; i <= SEGS; ++i) {
            float x, y;
            dialPoint(t0 + (t1 - t0) * i / SEGS, R, x, y);
            glVertex2f(x, y);
        }
        glEnd();
    }

    // Tick marks at the traditional VU positions
    const float ticks[] = { -20, -10, -7, -5, -3, -2, -1, 0, 1, 2, 3 };
    glLineWidth(2.0f);
    glBegin(GL_LINES);
    for (float vu : ticks) {
        float t = vuToDial(vu);
        if (vu > 0.0f) glColor4f(1.0f, 0.15f, 0.15f, 1.0f);
        else           glColor4f(0.75f, 0.75f, 0.8f, 1.0f);
        float len = (vu == 0.0f || vu == -20.0f || vu == 3.0f) ? 0.14f : 0.08f;
        float x0, y0, x1, y1;
        dialPoint(t, R, x0, y0);
        dialPoint(t, R * (1.0f - len), x1, y1);
        glVertex2f(x0, y0); glVertex2f(x1, y1);
    }
    glEnd();

    // Needle: warm white, turns red when it's in the red zone
    float nx, ny;
    dialPoint(needle, R * 1.04f, nx, ny);
    if (needle > zeroT) glColor4f(1.0f, 0.3f, 0.25f, 1.0f);
    else                glColor4f(1.0f, 0.92f, 0.75f, 1.0f);
    glLineWidth(2.5f);
    glBegin(GL_LINES);
        glVertex2f(cx, pivotY); glVertex2f(nx, ny);
    glEnd();

    // Pivot cap
    float cap = faceH * 0.04f;
    glColor4f(0.2f, 0.2f, 0.25f, 1.0f);
    glBegin(GL_QUADS);
        glVertex2f(cx - cap * xs, pivotY - cap); glVertex2f(cx + cap * xs, pivotY - cap);
        glVertex2f(cx + cap * xs, pivotY + cap); glVertex2f(cx - cap * xs, pivotY + cap);
    glEnd();

    // Peak LED in the top-right corner
    float led = faceH * 0.05f;
    float lx = right - faceW * 0.06f, ly = top - faceH * 0.1f;
    if (needle > zeroT) glColor4f(1.0f, 0.1f, 0.1f, 1.0f);
    else                glColor4f(0.25f, 0.05f, 0.05f, 1.0f);
    glBegin(GL_QUADS);
        glVertex2f(lx - led * xs, ly - led); glVertex2f(lx + led * xs, ly - led);
        glVertex2f(lx + led * xs, ly + led); glVertex2f(lx - led * xs, ly + led);
    glEnd();
}

void drawVUMeters() {
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    // Sit over the bar graph (which fills y = -1..0): left channel, right channel
    const float faceW = 0.62f, faceH = 0.48f, bottom = -0.55f;
    drawVUMeter(-0.45f, bottom, faceW, faceH, g_vuNeedleLeft);
    drawVUMeter( 0.45f, bottom, faceW, faceH, g_vuNeedleRight);

    glDisable(GL_BLEND);
}

// Loads the PNG into an OpenGL texture. Returns 0 on failure.
GLuint loadTexture(const char* path) {
    int w, h, n;
    unsigned char* data = stbi_load(path, &w, &h, &n, 4);
    if (!data) {
        std::cerr << "Could not load TV frame image '" << path << "': " << stbi_failure_reason() << "\n";
        return 0;
    }
    GLuint tex;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, data);
    stbi_image_free(data);
    return tex;
}

// Draws the TV frame stretched to the full window border, on top of everything.
void drawTVFrame() {
    if (!g_tvTexture) return;
    glEnable(GL_TEXTURE_2D);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glBindTexture(GL_TEXTURE_2D, g_tvTexture);
    glColor4f(1.0f, 1.0f, 1.0f, 1.0f);
    glBegin(GL_QUADS);   // image rows run top-down, so v = 0 is the top edge
        glTexCoord2f(0.0f, 1.0f); glVertex2f(-1.0f, -1.0f);
        glTexCoord2f(1.0f, 1.0f); glVertex2f( 1.0f, -1.0f);
        glTexCoord2f(1.0f, 0.0f); glVertex2f( 1.0f,  1.0f);
        glTexCoord2f(0.0f, 0.0f); glVertex2f(-1.0f,  1.0f);
    glEnd();
    glDisable(GL_BLEND);
    glDisable(GL_TEXTURE_2D);
}

int main(int argc, char** argv) {
    if (argc > 1) TV_FRAME_PATH = argv[1];

    // ---- 1. Initialize miniaudio context to enumerate devices ----
    ma_context context;
    if (ma_context_init(NULL, 0, NULL, &context) != MA_SUCCESS) {
        std::cerr << "Failed to initialize miniaudio context.\n";
        return -1;
    }

    ma_device_info* pCaptureDeviceInfos;
    ma_uint32 captureDeviceCount;
    if (ma_context_get_devices(&context, NULL, NULL, &pCaptureDeviceInfos, &captureDeviceCount) != MA_SUCCESS) {
        std::cerr << "Failed to retrieve audio devices.\n";
        ma_context_uninit(&context);
        return -1;
    }

    // Loop through capture devices to find the system's audio monitor
    ma_device_id* pSelectedDeviceID = NULL;
    std::cout << "Available Capture Devices:\n";
    for (ma_uint32 i = 0; i < captureDeviceCount; ++i) {
        std::string deviceName(pCaptureDeviceInfos[i].name);
        std::cout << " [" << i << "] " << deviceName << "\n";

        // Linux monitors usually contain "monitor", "Monitor", or "Loopback"
        if (deviceName.find("monitor") != std::string::npos || 
            deviceName.find("Monitor") != std::string::npos ||
            deviceName.find("Loopback") != std::string::npos) {
            pSelectedDeviceID = &pCaptureDeviceInfos[i].id;
            std::cout << "   --> Found and auto-selected desktop monitor!\n";
            break;
        }
    }

    // ---- 2. Initialize Audio Capture Device ----
    ma_device_config deviceConfig = ma_device_config_init(ma_device_type_capture); 
    deviceConfig.capture.format   = ma_format_f32; 
    deviceConfig.capture.channels = 2;             
    deviceConfig.sampleRate       = 44100;         
    deviceConfig.dataCallback     = audioCallback;
    
    // Explicitly target the monitor ID if found, otherwise falls back to default mic
    if (pSelectedDeviceID != NULL) {
        deviceConfig.capture.pDeviceID = pSelectedDeviceID;
    } else {
        std::cout << "No system monitor detected. Falling back to default microphone.\n";
    }

    ma_device device;
    // Pass the context explicitly as the first argument instead of NULL
    if (ma_device_init(&context, &deviceConfig, &device) != MA_SUCCESS) {
        std::cerr << "Failed to open audio device.\n";
        ma_context_uninit(&context);
        return -1;
    }

    if (ma_device_start(&device) != MA_SUCCESS) {
        std::cerr << "Failed to start audio listening engine.\n";
        ma_device_uninit(&device);
        ma_context_uninit(&context);
        return -1;
    }

    // ---- 3. Initialize Window & Context (GLFW) ----
    if (!glfwInit()) return -1;
    GLFWwindow* window = glfwCreateWindow(WINDOW_WIDTH, WINDOW_HEIGHT, "Auto-Routed Audio Visualizer", NULL, NULL);
    if (!window) {
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);

    g_tvTexture = loadTexture(TV_FRAME_PATH);

    glMatrixMode(GL_PROJECTION); glLoadIdentity();
    glMatrixMode(GL_MODELVIEW); glLoadIdentity();

    // ---- 4. Main Rendering Loop ----
    double lastTime = glfwGetTime();
    while (!glfwWindowShouldClose(window)) {
        double now = glfwGetTime();
        float dt = (float)(now - lastTime);
        lastTime = now;

        // Use the real framebuffer size so resizing / HiDPI screens work
        int fbW, fbH;
        glfwGetFramebufferSize(window, &fbW, &fbH);

        glViewport(0, 0, fbW, fbH);
        glClearColor(0.04f, 0.04f, 0.06f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        updateVUMeters(dt);

        // 1. Visualizer squeezed into the TV's screen area (or the whole window
        //    if the frame image couldn't be loaded)
        if (g_tvTexture) {
            int vx = (int)(SCREEN_LEFT * fbW);
            int vw = (int)(SCREEN_RIGHT * fbW) - vx;
            int vy = (int)((1.0f - SCREEN_BOTTOM) * fbH);   // GL origin is bottom-left
            int vh = (int)((1.0f - SCREEN_TOP) * fbH) - vy;
            glViewport(vx, vy, vw, vh);
            g_viewAspect = (float)vh / vw;
        } else {
            g_viewAspect = (float)fbH / fbW;
        }

        drawOscilloscope();
        drawEQGraph();
        drawVUMeters();   // drawn last so they sit on top of the bars

        // 2. TV frame over the full window
        glViewport(0, 0, fbW, fbH);
        drawTVFrame();

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    // Cleanup everything
    if (g_tvTexture) glDeleteTextures(1, &g_tvTexture);
    ma_device_uninit(&device);
    ma_context_uninit(&context);
    glfwTerminate();
    return 0;
}

