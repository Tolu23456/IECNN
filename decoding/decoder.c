#include <math.h>
#include <string.h>
#include <stdlib.h>

/* Fast image synthesis from latent (v6 SOTA) */
void render_image_fast(const float *latent, int width, int height, unsigned char *out_rgb) {
    // Uses 32 latent dimensions for spatial basis functions
    for (int r = 0; r < height; r++) {
        for (int c = 0; c < width; c++) {
            float r_acc = 0.0f, g_acc = 0.0f, b_acc = 0.0f;
            for (int k = 0; k < 8; k++) {
                float freq = (float)(k + 1);
                float basis = sinf((float)r * 0.1f * freq) * cosf((float)c * 0.1f * freq);
                r_acc += latent[k * 4] * basis;
                g_acc += latent[k * 4 + 1] * basis;
                b_acc += latent[k * 4 + 2] * basis;
            }
            int idx = (r * width + c) * 3;
            float r_out = r_acc * 127.0f + 128.0f;
            float g_out = g_acc * 127.0f + 128.0f;
            float b_out = b_acc * 127.0f + 128.0f;
            out_rgb[idx]     = (unsigned char)(r_out < 0 ? 0 : (r_out > 255 ? 255 : r_out));
            out_rgb[idx + 1] = (unsigned char)(g_out < 0 ? 0 : (g_out > 255 ? 255 : g_out));
            out_rgb[idx + 2] = (unsigned char)(b_out < 0 ? 0 : (b_out > 255 ? 255 : b_out));
        }
    }
}

/* Fast audio synthesis from latent (v6 SOTA) */
void render_audio_fast(const float *latent, int sr, float duration, short *out_pcm) {
    int n_samples = (int)(sr * duration);
    float pi2 = 2.0f * 3.1415926535f;

    for (int i = 0; i < n_samples; i++) {
        float t = (float)i / (float)sr;
        float val = 0.0f;
        // Use 16 latent dims as harmonic modulators
        for (int k = 0; k < 16; k++) {
            float freq = 110.0f * (k + 1);
            val += latent[k] * sinf(pi2 * freq * t);
        }
        val = tanhf(val);
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
