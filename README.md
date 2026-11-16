# 焦散透镜 RUN2

普通终端程序：单个完整 JSON → SDOT → JAX 自由曲面优化 → 独立正向追迹 → STL 和四张图。当前代码要求 Python 3.14；依赖版本记录在 requirements.txt。

## 运行

先自行激活已安装依赖的环境，再在 RUN2 目录运行。程序不选择或激活 Python/Mamba 安装，不修改用户配置：

```powershell
mamba activate caustic-jax
python -u design_lens.py --config configs/general_circle_full.json
```

快速检查和自定义输出目录：

```powershell
python -u design_lens.py --config configs/general_quick.json --out results/my_quick
```

输出目录必须不存在，可位于任意可写位置。JSON 的相对输入、输出和运行环境路径以该 JSON 所在目录为基准；CLI 相对路径以当前目录为基准。配置路径支持 ~ 和环境变量。程序无需 PowerShell 启动脚本；原本含有本机 Mamba/GCC 路径的 activate.ps1 已移除。

## 可选运行环境

所有环境配置集中在 runtime.py，只在导入 NumPy/JAX/SDOT 前执行一次。backend.py 不再设置设备、精度、缓存或 DLL 路径。

jax_enable_x64 默认 true，不需要写入 JSON；显式设置 false 可改用单精度。其他 runtime 项省略时不覆盖已有环境：设备、编译器和线程采用现有环境或库默认。缓存完全交给 SDOT/Loom/Matplotlib，不再提供项目缓存字段。以下示例选择 CPU 和全部可用 CPU，精度自动为双精度：

```json
"runtime": {
  "jax_platforms": "cpu",
  "threads": 0
}
```

| 可选字段 | 含义 |
|---|---|
| jax_platforms | 传给 JAX_PLATFORMS，例如 cpu；省略时不设置设备 |
| jax_enable_x64 | 默认 true，可省略；显式 false 关闭 JAX 双精度 |
| threads | 0 为当前进程可用 CPU 数；正整数指定线程数；省略时不覆盖线程环境 |
| xla_flags | 原样配置 XLA_FLAGS；若显式包含 {threads}，替换为本次线程数，且必须提供 threads |
| compiler | CXX 编译器路径或系统 PATH 中的命令名，例如 g++；省略时由系统/SDOT 选择 |
| path_dirs | 要前置到 PATH 的目录列表，使用当前系统路径分隔符 |
| dll_dirs | 显式注册的 DLL 搜索目录；配置后严格执行，当前 Python 缺少注册能力时直接报错，不静默忽略 |

runtime 只有一层，不按操作系统名选择或忽略配置。四个示例 JSON 直接指定当前使用的编译器、DLL、CPU、线程及编译器/DLL。compiler 未提供时沿用系统编译器；提供后必须可找到，否则报错，不自动回退。迁移到其他环境时修改或删除 compiler、path_dirs、dll_dirs 等显式机器路径；删掉整个 runtime 后，除默认开启 JAX 双精度外，其余沿用系统默认。

程序会在终端和 result.json 中报告实际设备、精度和环境。缓存不在运行 JSON 中指定，沿用已有 LOOM_CACHE_DIR/LOOM_BUILD_DIR/MPLCONFIGDIR 环境或库的系统用户缓存。Loom 自行加载生成 DLL，无需再配置项目 .cache/build/runtime 或 units。

## 物理与算法参数

- image / mask：正方形目标图和凸掩模。白色为透镜区域，透明区域按黑色处理。
- refractive_index / air_index：透镜和外部介质折射率。
- L_mm：整张掩模画布的毫米边长，不一定等于白色口径宽度。
- distance_mm：平面入光面 z=0 到固定屏幕的距离；追迹使用实际剩余距离 D-h。
- target_size_mm：整个目标正方形画布的投影边长，包括黑边。
- mesh_n / coarse_mesh_levels：基础网格间隔数和由粗到细的级别，最后自动加入 mesh_n。
- center_target：整数像素居中；mask_threshold：掩模白区阈值。
- optics：厚度范围、OT 站点、图像分辨率、迭代次数和损失权重。ot_resolution/image_resolution=0 为原图宽度，最小 16。
- stages：后续阶段列表；[] 表示无后续阶段。支持 name、adaptive、iterations、resolution、thin_weight、mapping_anchor、fine_samples。
- solver：本文件内完整指定 SDOT、LSMR、Adam、L-BFGS-B 参数，不引用外部预设。
- CLI --threads 可覆盖 runtime.threads，其余 CLI 覆盖对应物理/输入参数。

configs/general_high.json、general_balanced.json、general_quick.json、general_circle_full.json 均为互相独立的完整配置。未指定 --config 时读取脚本旁的 configs/general_high.json。

## 求解器

| 分组 | 参数 | 默认值/作用 |
|---|---|---|
| sdot | tol / max_iter | 1e-10 / 500，容差和每个密度阶段最大迭代 |
| sdot | density_stages | [0,0.5,0.9,0.99,0.999,1]；严格递增且从 0 到 1，最终黑区密度为零 |
| lsmr | atol / btol | 1e-9 / 1e-9，初始坡度积分容差 |
| lsmr | maxiter / conlim / damp | 3000 / 1e8 / 0 |
| adam | learning_rate / steps | 0.002 / 20，映射拟合的热启动；steps=0 关闭热启动 |
| adam | b1 / b2 / eps / eps_root / nesterov | 0.9 / 0.999 / 1e-8 / 0 / false |
| lbfgsb | ftol / gtol | 1e-12 / 1e-7，收敛容差 |
| lbfgsb | maxls / maxcor / maxfun | 35 / 20 / 15000 |

L-BFGS-B 最大迭代仍由 optics.fit_iterations、optics.image_iterations 或 stages[].iterations 控制，不在 solver 重复。

## 细化和结果选择

完整配置先运行 48→120→300→600 基础网格。adaptive 加密两轮，adaptive_fine 再一轮；detail、final 只优化高度。细线加密通过当前曲面折射落点选择面片，插入共享边中点并重新三角剖分。

只在无全反射候选中选择独立追迹 rmse_relative 最低者。无全反射要求实际追迹 tir=0 且全部面片 min_snell_discriminant>0。漏光、接收通量、细线比只记录，不作为通过阈值。

如果全部候选都有全反射，根目录仅输出 result.json，selection_status=no_tir_free_candidate，selected_stage/model/metrics 为 null；各阶段结果仍保留。

## 输出与日志

正常完成时，根目录和每个阶段各只有：

```text
lens.stl
height.png
comparison.png
loss.png
mapping.png
result.json
```

ascii_stl=true 时 lens.stl 为高精度文本 STL，false 为二进制 STL。height 使用 viridis，comparison 并排显示目标、独立模拟和绝对误差。根目录五个模型/图像来自选中的阶段，result.json 合并参数、全部指标、口径、环境、SDOT 和阶段记录。

数据直接在内存传递，不产生再删除 NPZ/CSV/OBJ/额外图片，不写日志文件。日志只输出到终端，格式 HH:MM:SS | 消息，无 PID、UI 或心跳线程。具体刀具补偿、刀路及抛光未验证，cnc_release=false。

## 原始 60 mm 圆形配置

general_circle_full.json：满口径 2048² 圆形掩模，n=1.5/1，L=60 mm，D=100 mm，目标画布 50 mm，基础网格最终 600，OT 站点 8192，后续优化使用原图宽度，独立验证 2^24 条光线。初始基准 9 mm，允许厚度 2–22 mm。像素凸包属于圆的栅格近似，面积约比理想圆大 0.109%。

此通用流程从头优化，与历史解析圆、多次续算和后期居中校正流程不同，不能保证历史曲面逐点复现。

## 可迁移性与验证

参见 PORTABILITY.md 的硬编码审计。已完成 Windows 小规模实际运行、系统默认/部分配置测试、实际 JAX 精度环境测试及“提供即执行、缺省即系统默认”测试。当前 WSL 访问被拒，未声称 Linux 已实跑 SDOT/JAX 全流程。依赖和本机 C++ 工具链仍需在目标系统正确安装。跨平台迁移只复制源码、configs、examples 和 requirements.txt，不复用 Windows 的 .cache 编译产物；由目标系统重新生成缓存。
