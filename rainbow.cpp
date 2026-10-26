
//g++ main.cpp -o live_visualizer -lglfw -lGL -lm -lpthread -ldl

#include <GLFW/glfw3.h>
#include <cmath>
#include <vector>
#include <iostream>
#include <complex>

// Define miniaudio implementation ONLY in one source file
#define MA_NO_JACK
#define MINIAUDIO_IMPLEMENTATION
#include "miniaudio.h"

// Constants
const int WINDOW_WIDTH = 800;
const int WINDOW_HEIGHT = 600;
const int BUFFER_SIZE = 512; // Must be a power of 2 for the FFT
const int EQ_BANDS = 32;

// Shared safe buffers between Audio thread and OpenGL thread
std::vector<float> g_waveformBuffer(BUFFER_SIZE, 0.0f);
std::vector<float> g_fftBands(EQ_BANDS, 0.0f);

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
int main() {
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

    glViewport(0, 0, WINDOW_WIDTH, WINDOW_HEIGHT);
    glMatrixMode(GL_PROJECTION); glLoadIdentity();
    glMatrixMode(GL_MODELVIEW); glLoadIdentity();

    // ---- 4. Main Rendering Loop ----
    while (!glfwWindowShouldClose(window)) {
        glClearColor(0.04f, 0.04f, 0.06f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        drawOscilloscope();
        drawEQGraph();

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    // Cleanup everything
    ma_device_uninit(&device);
    ma_context_uninit(&context);
    glfwTerminate();
    return 0;
}

