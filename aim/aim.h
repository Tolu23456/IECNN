#ifndef AIM_H
#define AIM_H

void invert_feature(const float *p, int n, float *out);
void invert_context(const float *p, int n, float *out);
void invert_spatial(const float *p, int n, float *out);
void invert_scale(const float *p, int n, float *out);
void invert_abstraction(const float *p, const float *ctx, int n, float *out);
void invert_noise(const float *p, int n, unsigned int seed, float *out);
void invert_relational(const float *p, int n, float *out);
void invert_temporal(const float *p, int n, float *out);
void invert_cross_modal(const float *p, int n, float *out);

#endif
