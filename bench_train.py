import time, sys
from pipeline.pipeline import IECNN
m = IECNN(persistence_path='global_brain.pkl', num_dots=64, n_heads=2, max_iterations=2)
with open('corpus_10k.txt') as f:
    lines = [f.readline().strip() for _ in range(200)]
m.fit(lines)
dots = m._ensure_dots()
for d in dots: d.n_heads = 1
m.aim.max_variants = 0
print('start train', flush=True)
t0 = time.time()
for i, s in enumerate(lines[:10], 1):
    t1=time.time()
    m.run(s, verbose=False)
    print(f'  sent {i}: {time.time()-t1:.2f}s', flush=True)
print(f'10 sents total {time.time()-t0:.2f}s', flush=True)
print('mean_eff:', m.dot_memory.summary()['mean_eff'], 'active:', m.dot_memory.summary()['active_dots'])
