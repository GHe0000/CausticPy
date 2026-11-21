# CausticPy

基于最优传输和可微物理的焦散透镜生成与优化工具，基于 Python + JAX 实现. 

项目包含下面两个部分：

- OT（最优输运，`run_OT.py`）：从目标图与掩模出发，从零生成焦散透镜模型
（参考：Yuliy Schwartzburg, Romain Testuz, Andrea Tagliasacchi, Mark Pauly. High-contrast Computational Caustic Design. ACM Transactions on Graphics (SIGGRAPH 2014).）


- DP（可微物理优化，`run_DP.py`）：读取已有 STL 透镜和目标图，通过可微物理的方式进行优化
（参考：Yuou Sun, Bailin Deng, Juyong Zhang. End-to-end Surface Optimization for Light Control. ACM Transactions on Graphics (arXiv:2408.13117).）


## 环境准备

1. Python 3.14 + 可用 C++ 编译器以及 Ninja

2. 安装对应库

```powershell
pip install -r requirements.txt
```

## 配置说明

在 `OT/config` 和 `DP/config` 中，有一些预设的 json 配置文件，运行前需要复制一份进行修改.


### OT 配置（`OT/config/*.json`）

常用参数：

- `image`：目标图像，必须是 1:1
- `mask`：掩模，必须是 1:1，白色为透光区域，透光区域必须是凸的
- `out`：默认输出目录（可被命令行 `--out` 覆盖）
- `refractive_index` / `air_index`：透镜与空气折射率
- `L_mm`：掩模对应边长
- `distance_mm`：投影距离，以相对透镜平面计算
- `target_size_mm`：目标图案对应宽度
- `mesh_n`：最终网格密度
- `coarse_mesh_levels`：细化过程，设为 `[]` 则只用 `mesh_n`
- `optics`：基准/最小/最大厚度（`base_mm`、`min_thickness_mm`、`max_thickness_mm`）、SDOT 点数（`ot_sites`）、拟合与图像迭代次数、损失权重、独立光追光线数（`2**trace_power`）
- `stages`：基础阶段之后的局部加密与精细优化阶段，设为 `[]` 可跳过细化

### DP 配置（`DP/config/default.json`）

常用参数：

- `input_stl`：初始 STL 路径
- `image`：目标图像，必须是 1:1
- `out`：默认输出目录（可被命令行 `--out` 覆盖）
- `optics`：物理参数（`refractive_index`、`air_index`、`screen_z_mm`、`target_size_mm`、`min_thickness_mm`、`max_thickness_mm`），需与初始 STL 一致
- `optimization`：L-BFGS 优化参数，如最大迭代次数 `iterations`（可用 `--iterations` 覆盖）、光栅化分块大小 `raster_chunk`
- `loss_weights`：各项损失权重（`image` 图像、`boundary` 越界、`smooth` 曲率平滑、`barrier` 防退化与防全反射）
- `validation`：独立光线追迹参数（光线数 `2**trace_power` 与随机种子 `seed`）

### 运行环境（`runtime`）

`runtime` 为可选配置项，不填则沿用系统默认：

- `jax_platforms`：计算后端（只测试过 CPU 计算，GPU 计算未测试过）
- `threads`：线程数，填 `0` 表示使用全部逻辑 CPU 核心
- `cpu_devices`：模拟的 JAX CPU 设备数（DP 用于并行光栅化）
- `compiler` / `path_dirs` / `dll_dirs`：OT 使用的 C++ 编译器及依赖目录，可直接删除（改用系统默认工具链），Linux/macOS 请勿填写 dll_dirs.

## 运行

OT 生成模型：

```bash
python -u .\run_OT.py --config .\OT\config\general_high.json --out .\OT\results\high_01
```

DP 细化模型：

```bash
python -u .\run_DP.py --config .\DP\config\default.json --iterations 20 --out .\DP\results\dp_01
```
