#ifndef CONVERGENCE_H
#define CONVERGENCE_H

int greedy_cluster(const float *preds, int n, int dim, float alpha, float threshold, int *assign);
float score_cluster(const float *preds, const float *confs, int n, int dim, float alpha);
void compute_centroid(const float *preds, int n, int dim, float *out);
float score_cluster_ultra(const float *preds, const float *confs, int n, int dim, float alpha, const float *repellent, float repellent_weight);

#endif
