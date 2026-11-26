# 附注：推理中的位置

训练时位置看起来只是“按顺序编号”，到推理时才变成一堆硬约束。这篇附注列五处会真的算错的地方，配套的数学推导见 [notes-rope.md](notes-rope.md)。

## 一、位置必须显式传入，不能从长度推

教学模型的 `apply_rope` 用 `torch.arange(x.shape[-2])` 生成位置，也就是“位置 = 这个张量里第几个 token”。全前缀计算时这恰好正确，因为每次都把 $0,1,\dots,T-1$ 整段重算一遍。

增量解码就不成立了：只喂一个新 token 时 $T=1$，`arange(1)` 只能给出位置 0，于是这个 token 被当成序列开头。[rope_math.py](rope_math.py) 用具体数字量出了代价：

| 情况 | Q 与 K 的点积 |
| --- | --- |
| 正确（Q@2 · K@1） | $1.851421$ |
| 被当成位置 0 | $0.148479$ |

这不是几个百分点的偏差，是完全不同的一次打分，而且不会抛出任何异常。

仓库里的推理路径已经改成显式传位置——[packed_attention.py:33](../../../../../zero2vllm/modeling/packed_attention.py#L33) 的函数签名直接要求它：

```python
def rope(x, positions, base=10000.0):
    """[tokens,heads,even_dim], using explicit request-local positions."""
```

位置从哪来也是显式的 offset，而不是数长度：

| 位置来源 | 代码 |
| --- | --- |
| 增量解码的起点 | [paged_runner.py:26](../../../../../zero2vllm/engine/paged_runner.py#L26) `offset = context.cached_positions` |
| 分块预填的位置段 | [kv_geometry.py:64](../../../../../zero2vllm/reference/kv_geometry.py#L64) `"position_offset": start, "positions": list(range(start, start+len(tokens)))` |
| 前缀已算部分的续接 | [execution.py:72](../../../../../zero2vllm/reference/execution.py#L72) `positions.extend(range(r.computed, end))` |

第二行是最容易踩的：**chunked prefill**（把长 prompt 分块预填）时，第二块的位置必须接着第一块，不能从 0 重来。

## 二、KV cache 之所以能复用，是因为 K 的旋转只依赖它自己的位置

这一点常被想反。缓存里的 $K_j$ 是按它**出生时**的位置 $j$ 旋转的，之后永久有效——因为新 token 在位置 $p$ 打分时需要的只是 $(j-p)$，而 $j$ 与 $p$ 都是绝对值，差值稳定。

所以“cache 会不会因为新 token 位置更大而失效”——**不会**。只要每个 K 按自己的位置旋转一次，之后不必再动。

反过来有两条硬约束：

- **换位置方案，缓存就作废。** 做位置插值（压缩 $p$）或 NTK 缩放（放大 base）时，缓存里的 K 是按旧方案旋转的，新旧 token 的位置语义不再一致，必须整段重算，或按统一方案重新处理。
- **prefix caching 能成立，前提是位置可复现。** 相同前缀意味着相同位置、相同旋转，所以缓存可以跨请求复用；如果位置依赖请求上下文而变，复用的语义就没了。

## 三、批处理与打包：位置是“每请求本地”的

打包多条请求进一行时，每条请求的位置都从自己的 0 开始。仓库里这条注释写得很明确：

```python
# Request-local absolute position; not row or chunk-local index.
visible = torch.arange(length) <= batch.positions[row]
```

如果把两条请求的位置串成 $0,1,2,\dots$ 连续编号，第二条请求的 token 之间就会凭空多出一段距离，RoPE 按这个假距离去调制分数——注意力会以为它们在序列里隔得很远。

padding 位置通常填 0（[execution.py:84](../../../../../zero2vllm/reference/execution.py#L84) `positions.extend([0]*(padded-actual))`），但必须配合 mask 屏蔽，否则这些位置会被真的读进去。`packed_attention.py` 在两处直接抛错来防这类问题：`Attempt to overwrite an already written position`（[:111](../../../../../zero2vllm/modeling/packed_attention.py#L111)）与 `Attention tried to read an unwritten position`（[:133](../../../../../zero2vllm/modeling/packed_attention.py#L133)）。

## 四、上下文长度的真上限由最慢那个频率决定

$\text{最慢波长}=2\pi\times\text{base}\approx62832$（base$=10^4$）。超过它之后所有对的相位都绕过一圈以上，组合方式是训练时没见过的——**这不是显存问题，是位置语义失效**。

所以判断一个模型能不能吃下某个长度，看的是 $2\pi\times\text{base}$ 与训练长度，不是看显存够不够。推理端的应对也都落在“改频率/改位置”上：放大 base（NTK-aware）、压缩 $p$（位置插值）、两者混合（YaRN）。

## 五、配对约定必须与 checkpoint 一致

这一条最阴，因为它既合法、又不报错。仓库里两种约定并存：

| 实现 | 配对方式 | 代码特征 |
| --- | --- | --- |
| [torch_decoder.apply_rope](../../../../../zero2vllm/modeling/torch_decoder.py#L29)（教学模型） | **相邻**：$(0,1),(2,3),\dots$ | `x[..., 0::2]`, `x[..., 1::2]` 后 `stack(...).flatten(-2)` |
| [llama.split_half_rope](../../../../../zero2vllm/modeling/llama.py#L74) | **前后半区**：$(0,\frac{D_h}{2}),(1,\frac{D_h}{2}+1),\dots$ | `x.chunk(2, -1)` 后 `cat((-second, first))` |
| [packed_attention.rope](../../../../../zero2vllm/modeling/packed_attention.py#L33) | 同半区 | 同上 |

`llama.py:75` 的 docstring 直接写了 `Rotate (0, Dh/2), (1, Dh/2+1), ...`，而教学模型用的是相邻配对。两者**数学上只差一个固定的维度置换**，频率表 $\omega_i$ 完全相同，差别只在于“哪两个分量配成一对”：

```text
Dh=4, 向量 [a0, a1, a2, a3]
相邻配对:   (a0,a1) 用 ω0    (a2,a3) 用 ω1
半区配对:   (a0,a2) 用 ω0    (a1,a3) 用 ω1
```

用错约定时，$a_1$ 会被按 $\omega_0$ 而不是 $\omega_1$ 旋转——**每个位置的每个角度都是错的**，但形状合法、不产生 `nan`、不做任何检查，只是输出质量莫名其妙地变差。这类错误在 trace 里表现为“logits 全都不对”，很难定位到 RoPE 本身。

因此加载真实 checkpoint 时，配对约定是最该先核对的一行代码，比 dtype、比归一化位置都隐蔽。判断方法很直接：看它是 `0::2` 风格还是 `chunk(2)` 风格，再与权重发布方的配置对齐。

## 小结

| 场景 | 该做的事 | 做错的后果 |
| --- | --- | --- |
| 增量解码 / chunked prefill | 显式传绝对位置（或 offset） | 新 token 被当成位置 0，打分全错 |
| KV cache 复用 | 每个 K 按出生位置旋转一次 | 换位置方案后缓存语义失效 |
| 多请求打包 | 位置按请求各自从 0 起 | 假距离调制分数 |
| 超长上下文 | 核对 $2\pi\times\text{base}$，必要时改 base 或压缩 $p$ | 位置语义失效，质量下降 |
| 加载 checkpoint | 核对相邻配对还是半区配对 | 角度全错且不报错 |

[返回示例索引](README.md) · [返回正文](../README.md)
