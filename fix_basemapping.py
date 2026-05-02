import re
import os

path = 'basemapping/basemapping.py'
with open(path, 'r') as f:
    content = f.read()

# 1. Deterministic hashing
content = re.sub(
    r'def _stable_embedding\(token: str, dim: int\) -> np\.ndarray:\n    """Stable hash-based unit-sphere embedding for a token string\."""\n    seed = abs\(hash\(token\)\) % \(2 \*\* 31\)\n    rng = np\.random\.RandomState\(seed\)\n    v = rng\.randn\(dim\)\.astype\(np\.float32\)\n    n = np\.linalg\.norm\(v\)\n    return v / n if n > 1e-10 else v',
    'def _stable_embedding(token: str, dim: int) -> np.ndarray:\n    """Stable hash-based unit-sphere embedding for a token string (v6 deterministic)."""\n    import hashlib\n    h = hashlib.sha256(token.encode("utf-8")).hexdigest()\n    seed = int(h[:8], 16)\n    rng = np.random.RandomState(seed)\n    v = rng.randn(dim).astype(np.float32)\n    n = np.linalg.norm(v) + 1e-10\n    return v / n',
    content
)

# 2. Correct video averaging axes
content = content.replace(
    'block = frame.reshape(8, 8, 8, 8, 3).mean(axis=(2, 4)).flatten()',
    '# Corrected spatial axes for V6 SOTA\n            block = frame.reshape(8, 8, 8, 8, 3).mean(axis=(1, 3)).flatten()'
)

# 3. Normalized context similarity
content = content.replace(
    'sim_matrix = embeddings @ embeddings.T',
    '# Ensure embeddings are normalized for similarity (v6 stable)\n        norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-10\n        embeddings_norm = embeddings / norms\n        sim_matrix = embeddings_norm @ embeddings_norm.T'
)

# 4. Memory-safe AAF
content = content.replace(
    'if n > 256:',
    'if n > 128:'
)
content = content.replace(
    'window = 64',
    'window = 32'
)

# 5. Composite matching refinement
content = re.sub(
    r'# 0\. Composite Concepts\s+for comp_name, comp_type in self\._base_types\.items\(\):\s+if comp_type == "composite":\s+# Check if next tokens match the name\s+name_parts = comp_name\.split\("_"\) # e\.g\. .concept_123.\s+# Simplified check: just look for the literal name if it.s a single token\s+if comp_name == tokens\[i\]:\s+result_toks\.append\(comp_name\)\s+result_types\.append\("composite"\)\s+i \+= 1\s+matched = True\s+break',
    '# 0. Composite Concepts (Semantic Activation Matching)\n                curr_emb = self._token_embedding(tokens[i], "word")\n                for name, emb in self._base_vocab.items():\n                    if self._base_types.get(name) == "composite":\n                        sim = float(np.dot(curr_emb, emb))\n                        if sim > 0.92:\n                            result_toks.append(name)\n                            result_types.append("composite")\n                            i += 1\n                            matched = True\n                            break',
    content
)

with open(path, 'w') as f:
    f.write(content)
print("Applied fixes via script.")
