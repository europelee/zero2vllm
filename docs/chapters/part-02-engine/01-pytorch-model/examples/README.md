# 本节调用示例

本目录的调用示例复用仓库中的共享模型，不复制模型实现。先按 [实验说明](../lab.md) 安装 CPU PyTorch 环境，命令在仓库根目录运行。

| 示例 | 输入 | 阅读目的 |
| --- | --- | --- |
| [attention_math.py](attention_math.py) | 手工指定的 Q、K、V | 复现正文三位置 Attention 手算，观察正确和错误 mask 的差别 |
| [rope_math.py](rope_math.py) | 两个头各自的四维向量 | 逐对核对旋转、长度保持、相对位置关系，演示单 token 位置重置错误 |
| [walk_forward.py](walk_forward.py) | `[[1,2,3]]` | 逐层运行模型，观察形状和最后位置的预测 |
| [run.py](run.py) | `[[1,2,3,4]]` | 查看完整 trace 与四步生成，使用默认实验输入 |

```bash
python docs/chapters/part-02-engine/01-pytorch-model/examples/attention_math.py
python docs/chapters/part-02-engine/01-pytorch-model/examples/rope_math.py
python docs/chapters/part-02-engine/01-pytorch-model/examples/walk_forward.py
python docs/chapters/part-02-engine/01-pytorch-model/examples/run.py
```

PowerShell 未激活环境时，用 `.\.venv\Scripts\python.exe` 替换 `python`。`attention_math.py` 与 `rope_math.py` 使用各自独立的手算数据；另外两个示例的 ID 来自随机教学权重。它们的输入不同，不应比较它们是否生成相同的 ID。

## 延伸阅读

脚本里有几行代码的“为什么”超出了本节正文的范围，单独记在下面几篇附注里。它们只讨论数值与工程细节，不重复正文的推导。

| 附注 | 回答的问题 |
| --- | --- |
| [notes-softmax.md](notes-softmax.md) | `softmax(dim=-1)` 为什么要先减最大值：平移不变性是许可而非累赘、fp16/fp32 的溢出与归零阈值、为什么这不是“多算一遍” |
| [notes-rope.md](notes-rope.md) | 从注意力公式到频率阶梯：注意力为什么没有位置、$R(i\omega)^{\mathsf T}R(j\omega)=R((j-i)\omega)$ 怎么推出来、位置写进 q/k 后如何生效、频率为什么取 $p\cdot\text{base}^{-2m/D_h}$ |
| [notes-rope-inference.md](notes-rope-inference.md) | 位置在推理时的约束：何时必须显式传位置、KV cache 为什么能复用、多请求打包、上下文长度的真上限、配对约定与 checkpoint 的对应 |

几篇附注里的数字都在本机 CPU PyTorch 环境实测得到。与脚本输出对应的部分（如 RoPE 的逐对角度、位置重置前后的点积）可以直接用上面的命令复现；溢出阈值、拟合残差与耗时这类测量则在附注中写明了测量方法。

[返回正文](../README.md)
