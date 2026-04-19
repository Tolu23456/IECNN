#ifndef AIM_H
#define AIM_H

/*
 * AIM (Attention Inverse Mechanism) — C Implementation
 * Six inversion types that challenge dot prediction assumptions.
 */

/* Feature Inversion: negate dominant attribute dimensions */
void invert_feature(const float *p, int n, float *out);

/* Context Inversion: swap high/low activation groups (role reversal) */
void invert_context(const float *p, int n, float *out);

/* Spatial Inversion: reverse dimension group ordering */
void invert_spatial(const float *p, int n, float *out);

/* Scale Inversion: rescale groups by inverse factors */
void invert_scale(const float *p, int n, float *out);

/* Abstraction Inversion: flip between levels of understanding */
void invert_abstraction(const float *p, const float *ctx, int n, float *out);

/* Noise/Absence Inversion: suppress dominant features */
void invert_noise(const float *p, int n, unsigned int seed, float *out);

#endif
