#include <math.h>
#include <string.h>
#include <stdlib.h>

/* Fast image synthesis from latent */
void render_image_fast(const float *latent, int width, int height, unsigned char *out_rgb) {
    float r_v = latent[0] * 127.0f + 128.0f;
    float g_v = latent[1] * 127.0f + 128.0f;
    float b_v = latent[2] * 127.0f + 128.0f;
    float l3 = latent[3];
    float l4 = latent[4];

    for (int r = 0; r < height; r++) {
        for (int c = 0; c < width; c++) {
            float v = (sinf((float)r / 10.0f * l3) + cosf((float)c / 10.0f * l4)) * 0.5f + 0.5f;
            int idx = (r * width + c) * 3;
            float r_out = r_v * v;
            float g_out = g_v * v;
            float b_out = b_v * v;
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
