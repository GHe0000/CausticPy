# 焦散透镜运行文件

使用已有 Windows / Mamba 环境 caustic-jax（Python 3.14）。只提供终端入口，结果保存到 results，不弹出窗口。

## 运行

PowerShell 进入本文件夹，每条命令为一行：
```powershell
Set-Location 'D:\Code\CausticDP\JAX3\RUN'
. .\activate.ps1
python -u design_lens.py --config configs/general_high.json
```

activate.ps1 激活 caustic-jax 并配置本机 SDOT 的 GCC/DLL 环境。当前已验证编译器位于 D:\Software\msys2\ucrt64\bin\g++.exe，Mamba 位于 D:\Software\miniforge3；迁移机器时需修改激活脚本和原 backend.py 中的本机路径。本包不包含 Python 环境或编译器。requirements.txt 记录当前主要依赖版本，不代表跨机器安装已经验证。

快速检查（32 网格，低质量，仅验证流程）：
```powershell
python -u design_lens.py --config configs/general_quick.json
```

指定其他结果文件夹：
```powershell
python -u design_lens.py --config configs/general_high.json --out results/my_lens
```

输出目录必须尚不存在，且位于 RUN 内。首次调用 SDOT 时会在 .cache 编译本机扩展，耗时可能较长。

## 配置

编辑 configs/general_high.json。image、mask、out 的相对路径均相对于该 JSON；CLI 指定的路径相对于当前目录。

- image：目标灰度 PNG，必须为正方形。
- mask：白色为透镜区域、黑色为非透镜区域，必须为正方形且白色区域为凸形。
- L_mm：整张掩模画布对应的物理边长，默认 60 mm。
- refractive_index / air_index：透镜 / 空气折射率，默认 1.5 / 1。
- distance_mm：从平面入光面 z=0 到屏幕的距离，默认 100 mm；出光高度 h 使用实际剩余距离 D-h。
- target_size_mm：屏幕上整张目标画布边长，默认 50 mm。
- mesh_n：覆盖掩模画布的初始间隔数，默认 600，初始间距 L/600；凸口径裁切和局部加密会改变总顶点数。
- coarse_mesh_levels：基础网格分级，例如 [48, 120, 300]，最后自动加入 mesh_n；[] 表示只用 mesh_n。
- threads：0 表示使用全部逻辑核心。
- center_target：通过整数像素平移居中，不插值。
- mask_threshold：灰度大于该值视为白，默认 127。
- optics：本次运行的全部光学/优化参数。ot_resolution、image_resolution 为显式分辨率，0 表示使用输入目标图宽度，最小值为 16；正数不再根据输入尺寸自动截小。
- stages：本次运行的后续优化阶段列表；[] 不执行后续阶段。每个阶段可设置 name、adaptive、iterations、resolution、thin_weight、mapping_anchor、fine_samples。

configs/general_high.json、general_balanced.json、general_quick.json 各自包含完整运行参数，互不引用。不指定 --config 时直接读取脚本旁的 configs/general_high.json。复制一个完整 JSON 即可配置新任务，无需修改 Python。

## 曲面细化

high 的基础网格依次为 48、120、300、mesh_n（默认 600）。旧曲面的高度插值到新网格，再通过真实 Snell 折射优化。

随后 adaptive 阶段执行 2 轮局部加密，adaptive_fine 再执行 1 轮。使用当前曲面折射后的落点，选择细线附近面片，在共享边插入中点、重新三角剖分，然后继续优化高度。这会实际增加三角网格顶点。detail 和 final 阶段只继续优化高度，不再插点。

general_balanced.json 局部加密 1 轮；general_quick.json 为检验完整流程指定了 1 轮小规模加密。各文件的 stages 就是实际执行的阶段。

本方法仍为分片平面三角网格高度场，不输出 NURBS。网格过粗仍会影响法向和成像；局部加密不保证任意输入都达到高质量。

## 流程和输出

灰度目标与凸掩模 → SDOT 能量映射 → JAX/Optax 多级自由曲面优化 → 局部加密与细线优化 → 独立 NumPy 几何光线追迹 → Trimesh 实体检查和 STL/OBJ 导出。

终端打印阶段、网格数量、优化损失和追迹进度，无心跳线程。使用 CPU、JAX float64。Matplotlib 只保存图片，高度色图固定为 viridis。

results/general_high/ 根目录包含：

- lens.stl、lens_precision.stl、lens.obj。
- mesh.png、overview.png、height.png。
- target.png、simulated.png、error.png、comparison.png。
- loss.png、loss.csv、mapping.png。
- settings.json、config.json、workflow.json、metrics.json、result.json、run.log 等记录。

base/ 和各优化阶段保留独立模型、检查点及验证记录。最终模型由独立追迹结果选出。pipeline_complete 只表示流程完成，光学质量需查看 optical_quality_pass 及实际模拟图。STL 按 mm 导入 CAM；具体刀具半径、加工步距和抛光未验证，cnc_release=false。

caustic/ 保留原已验证计算模块，逐字节复制；未复制旧平移校正模块、历史脚本、测试、参考项目、缓存或历史结果。

## 原始 60 mm 圆形透镜的完整计算

使用 configs/general_circle_full.json；它包含全部参数，不引用其他配置。对应命令：

```powershell
python -u design_lens.py --config configs/general_circle_full.json
```

折射率 1.5/1，口径横纵包围尺寸 60 mm，屏幕 z=100 mm，目标画布 50 mm，网格 48→120→300→600，8192 个 OT 站点，OT 使用原图宽度，基础图像优化 384，后续阶段 resolution=0 使用原图宽度，独立验证 2^24 条光线。厚度范围 2–22 mm、初始基准 9 mm；fine_quadrature=0 与历史圆形配置一致，仍保留基础面片采样。输出 results/circle60_full/。

mask_circle_full.png 为 2048×2048 满口径圆掩模；与旧的带黑边示例不同，L=60 时横纵尺寸均为 60 mm。轮廓仍是像素方格凸包，属于圆的栅格近似，已测面积比理想圆大约 0.109%。

此配置沿用原物理和主要精度参数，由通用入口从头优化。历史最佳 saturn_centered_poly 使用旧的解析圆口径与多次续算、后期居中校正，流程不同，不能保证曲面或质量指标完全相同。这里只核对了新配置和口径，尚未启动该耗时完整计算。
