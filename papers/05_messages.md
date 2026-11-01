# 05. Messages (Optional Dot Communication)

Messages let nearby dots in the same region tell each other something richer than "how I scored". This is a switchable extension, not part of the baseline.

Status: draft v0.1. Decision D9: build the thin channel first (weights plus pooled opinion), add messages later as an on/off switch.

---

## 1. Why messages exist

The thin channel tells a dot how it scored against the crowd, but not **why**. A dot cannot say "I think this is the start of a name, someone check the capital letter". Messages give dots a small side channel for that kind of coordination.

They also push IECNN further from a plain ensemble, which is why they need their own ablation: if messages do not help, they must be removed.

## 2. Rules

1. **Same region only.** Dots talk to dots in their own region (decided).
2. **Small bandwidth.** Messages have dimension `d_m` (default 16), much smaller than the dot state, so dots cannot just hand over their whole answer.
3. **One pass delay.** A message written at pass p is read at pass p+1. This keeps each pass parallel (no same-pass dependencies).
4. **Nearby only.** Each dot reads from a limited neighbourhood, not from all dots.
5. **A switch.** `use_messages: false` must give exactly the baseline model.

## 3. Mechanism

```
write:   m_k^p = tanh( W_m s_k^p )                         # [d_m], from the dot's updated state

neighbours:  N(k)   # topology below, never includes k itself

read:    a_kj = softmax_{j in N(k)} ( (Wq m_k^p) . (Wk m_j^p) / sqrt(d_m) )
         m_in_k^{p+1} = sum_{j in N(k)} a_kj * m_j^p       # [d_m]
```

`m_in` is concatenated into the GRU input at the next pass (file 01, input table).

### Neighbourhood topologies to compare

| Topology | Description | Cost |
|---|---|---|
| Ring | Dots arranged in a ring; each reads from the s nearest on each side (default s = 2) | Lowest |
| Full region | Every dot reads from all others in the region | Higher, and a copying risk |
| Learned top-k | Each dot attends over all others, keeps its top k scores | Highest |

Default: **ring with s = 2**.

## 4. Preventing copying

Messages must not become a route for dots to copy each other.

- **Small `d_m`** limits bandwidth.
- **Message dropout**: zero a random fraction of messages in training (default 0.2).
- **Noise**: add small Gaussian noise to messages in training.
- **No distributions in messages**: messages come from a projection of the state, not from `pi_k`.
- Monitor the pairwise error correlation with messages on and off (file 08). If correlation rises when messages are on, the channel is encouraging copying.

## 5. Parameters

Per dot: `W_m` of shape `[d_s, d_m]`. Shared in a region: `Wq`, `Wk` of shape `[d_m, d_m]`. Small compared with the GRU.

## 6. Diagnostics

| Diagnostic | How | Meaning |
|---|---|---|
| Message influence | KL between a dot's output with its incoming message vs with the message zeroed | Whether the receiver actually uses the message |
| Attention entropy over neighbours | Mean entropy of `a_kj` | Collapsing to one neighbour vs spreading |
| Error correlation delta | Pairwise error correlation with messages on minus off | Copying pressure |
| Downstream gain | Loss with vs without messages | The headline result (rung E3) |

## 7. Failure modes

| Failure | Mitigation |
|---|---|
| Messages ignored (influence near zero) | The switch is useless; remove it. |
| Messages cause copying | Lower `d_m`, raise dropout, shrink the neighbourhood. |
| Training instability from feedback loops | One-pass delay, gradient clipping, LayerNorm on messages. |
| Extra cost without gain | Cut at rung E3 if it fails the criterion in file 08. |

## 8. Tests

1. With `use_messages: false`, outputs are identical to the baseline for the same weights and seed.
2. A dot never reads its own message.
3. Messages written at pass p cannot affect outputs at pass p (one-pass delay).
4. The ring topology wraps correctly for the first and last dots.
5. Zeroing all messages with `use_messages: true` matches the no-message model within tolerance.
