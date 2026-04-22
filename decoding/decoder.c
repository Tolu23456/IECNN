#include <math.h>
#include <string.h>
#include <stdlib.h>

/* Upgraded Patch-Based Neural Rendering
 * Uses 32 latent dimensions to drive a set of spatial basis functions.
 */
void render_image_fast(const float *latent, int width, int height, unsigned char *out_rgb) {
    /* Latent dims 0-2: Base RGB */
    float base_r = latent[0] * 127.0f + 128.0f;
    float base_g = latent[1] * 127.0f + 128.0f;
    float base_b = latent[2] * 127.0f + 128.0f;

    for (int r = 0; r < height; r++) {
        for (int c = 0; c < width; c++) {
            float val = 0.0f;
            float rel_r = (float)r / (float)height;
            float rel_c = (float)c / (float)width;

            /* Accumulate from 8 spatial basis functions (dims 8-15) */
            for (int i = 0; i < 8; i++) {
                float freq_r = (float)(i + 1);
                float freq_c = (float)(8 - i);
                float phase = latent[16 + i] * 3.14159f;
                float amp = latent[8 + i];
                val += amp * sinf(rel_r * 6.28f * freq_r + rel_c * 6.28f * freq_c + phase);
            }

            /* Add high-frequency texture (dims 24-31) */
            for (int i = 0; i < 8; i++) {
                float tx = sinf(rel_r * 50.0f * (float)(i+1)) * cosf(rel_c * 50.0f * (float)(8-i));
                val += 0.2f * latent[24 + i] * tx;
            }

            float v = 0.5f + 0.5f * tanhf(val);
            int idx = (r * width + c) * 3;

            float r_out = base_r * v + latent[3] * 50.0f;
            float g_out = base_g * v + latent[4] * 50.0f;
            float b_out = base_b * v + latent[5] * 50.0f;

            out_rgb[idx]     = (unsigned char)(r_out < 0 ? 0 : (r_out > 255 ? 255 : r_out));
            out_rgb[idx + 1] = (unsigned char)(g_out < 0 ? 0 : (g_out > 255 ? 255 : g_out));
            out_rgb[idx + 2] = (unsigned char)(b_out < 0 ? 0 : (b_out > 255 ? 255 : b_out));
        }
    }
}

/* Fast audio synthesis from latent */
void render_audio_fast(const float *latent, int sr, float duration, short *out_pcm) {
    int n_samples = (int)(sr * duration);
    float f1 = 200.0f + fabsf(latent[0]) * 400.0f;
    float f2 = 400.0f + fabsf(latent[1]) * 600.0f;
    float pi2 = 2.0f * 3.1415926535f;

    for (int i = 0; i < n_samples; i++) {
        float t = (float)i / (float)sr;
        float val = 0.5f * sinf(pi2 * f1 * t) + 0.5f * sinf(pi2 * f2 * t);
        out_pcm[i] = (short)(val * 32767.0f);
    }
}

/* Temporal Continuity for Video Generation
 * Updates current frame based on previous frame and motion latent.
 */
void render_video_frame_fast(const float *prev_frame, const float *motion_latent,
                            int width, int height, float alpha, unsigned char *out_rgb) {
    for (int i = 0; i < width * height * 3; i++) {
        float base = (float)prev_frame[i];
        float shift = motion_latent[i % 256] * 10.0f; // Scale motion
        float val = base + shift;
        out_rgb[i] = (unsigned char)(val < 0 ? 0 : (val > 255 ? 255 : val));
    }
}
