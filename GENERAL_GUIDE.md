# 通用凸掩模焦散透镜程序

入口：`run_design.ps1` / `design_lens.py`。复用已经实跑的 SDOT + JAX/Optax + Trimesh 实现。按本任务约束只支持凸口径，使用现有 NumPy/SciPy，不引入 Gmsh、孔洞处理或额外 UI。结果只保存到指定文件夹，不调用 plt.show()。

## 1. 输入、尺寸和物理约定

- 目标图、掩模分别必须是 1:1 正方形，像素数可以不同。非正方形直接报错，不自动拉伸、裁切或补边。
- 掩模白色是透镜，黑色是非透镜；默认灰度 >127 为白。RGBA 透明部分按黑色处理。输入保证凸形，对白像素方格的边界角点取凸包作为机械轮廓，不支持凹口、孔洞或多片透镜。
- `L_mm` 是整张掩模画布的物理边长，不一定是白色轮廓的宽度/直径。N×N 掩模的像素边长为 L/N mm。
- 第 i 行、第 j 列的像素对应 x∈[-L/2+jL/N, -L/2+(j+1)L/N]，y∈[L/2-(i+1)L/N, L/2-iL/N]。原点为掩模画布中心，x 向右、y 向上；不自动平移掩模。
- `target_size_mm` 是屏幕上整张目标正方形画布的边长，含黑边，独立于 L。目标图案默认用整数像素平移居中，不插值、不丢非零像素。
- 透镜平面入光面 z=0，自由曲面 z=h(x,y)，均匀平行光沿 +z。屏幕固定 z=D。`--distance` / `--focal-length` 均表示从平面入光面到屏幕的 D，不是从最高点量起，也不是另一个独立的近轴光学焦距。
- 光通量定义在透镜凸口径内。实际光束应匹配透镜口径或遮挡黑色非透镜区域；从非透镜区域直接穿过空气的光不属于这里设计的入射光。
- 二值像素边界有栅格误差，凸包可能增加少量边缘面积。domain.json 记录白像素面积、凸包面积和差值；SDOT、网格和追迹统一使用凸包。

原参考图为 1227×1080，通用入口会拒绝。已显式生成 examples/target_square.png（1227×1227），仅补黑边，原文件不改动。辅助命令：
```powershell
python make_square_inputs.py --image target_saturn.png --out examples
```

## 2. 一条 PowerShell 命令运行

下面每条命令均为单行，输出目录必须不存在：
```powershell
& 'D:\Code\CausticDP\JAX3\run_design.ps1' --image examples/target_square.png --mask examples/mask_circle.png --n 1.5 --L 60 --distance 100 --target-size 50 --grid 600 --profile high --threads 0 --out results/my_lens
```

包装脚本自动进入 JAX3，执行 mamba activate caustic-jax，再启动 Python。默认 CPU、float64，threads=0 使用全部逻辑核心。已经激活环境时可直接：
```powershell
python -u design_lens.py --image examples/target_square.png --mask examples/mask_convex.png --n 1.45 --L 40 --focal-length 120 --target-size 30 --grid 32 --profile quick --out results/my_quick
```

统一 JSON 配置：
```powershell
& 'D:\Code\CausticDP\JAX3\run_design.ps1' --config configs/general_high.json --out results/my_high
```

JSON 的相对路径以配置文件目录为基准，命令行路径以当前目录为基准；命令行覆盖 JSON。所有输出限制在 JAX3。

## 3. 参数

| 参数 | CLI | 含义/默认值 |
|---|---|---|
| image | --image / --target | 必填，正方形目标 PNG |
| mask | --mask | 必填，正方形凸掩模 |
| refractive_index | --n | 透镜折射率，1.5 |
| air_index | --n-air | 外部介质折射率，1.0 |
| L_mm | --L | 掩模画布边长，60 mm |
| distance_mm | --distance / --focal-length | z=0 到屏幕的距离，100 mm |
| target_size_mm | --target-size | 目标画布边长，50 mm |
| mesh_n | --grid / --mesh-n | 每 L 划分间隔数，600，初始间距约 L/mesh_n |
| profile | --profile | quick / balanced / high，默认 high |
| threads | --threads | 0 为全部逻辑核心，也可指定正整数 |
| center_target | --no-center 关闭 | 默认整数像素居中图案 |
| out | --out | 必填，新输出目录 |

`grid=600` 不是只有 600 个顶点，也不保证总数恰好 600²。它规定覆盖整张掩模画布的初始间距；只保留凸口径内点，并加密边界。细线局部加密会继续增加顶点，实际数量逐级打印。

高级参数集中在 JSON 的 optics 字典：ot_sites、ot_resolution、ot_tol、fit_iterations、image_iterations、trace_power、fine_quadrature、厚度范围和损失权重。通用入口中 ot_resolution=0 表示原图宽度。默认机械范围 min_thickness_mm=2、base_mm=9、max_thickness_mm=22，必须满足 0<min<base<max<D；投影距离很短时应同时设置适当厚度。

stages 为后续优化阶段列表，支持 name、adaptive（局部加密轮数）、iterations、resolution（0 为原图）、thin_weight、mapping_anchor、fine_samples。stages=[] 只运行基础阶段。物理参数放顶层，不在 optics 重复定义。

## 4. 实现与完整流程

### 4.1 凸掩模 → 物理网格

`domain.py` 将像素边界换成 mm，SciPy ConvexHull 得到凸域 Ω 和半空间。内部规则点及边界细分点作 Delaunay 剖分。凸域假设保证不跨越真实凹口/孔洞。斜边上浮点误差可能产生近共线薄三角形，clean_convex_mesh 清理数值退化面片，重建单一边界，校验面积。

### 4.2 SDOT 能量对应

均匀源密度 ρs=1/Area(Ω)。Sobol 点在凸域内按面积采样作为站点，SDOT 对源 Voronoi 单元积分，得到 ai=Area(Vi∩Ω)/Area(Ω)，Σai=1，不假设每个站点等质量。

目标灰度 Iij 对应单位总光通量 tij=Iij/ΣIij。目标是二维分段常量像素密度。SDOT 求 Laguerre/Power Diagram 权重，使目标单元质量等于源容量；源/目标光通量重心建立对应，再作分片线性插值，边界必要时最近邻外推。

零密度黑区需要可行初值，因此求解 Iα=αI+(1-α)mean(I)，α 从 0 逐渐到 1。最后一级严格为原目标，黑色仍为零，没有最终灰度下限。每级及最终残差保存并打印。这里调用成熟 SDOT 连续测度积分，不是离散 Sinkhorn。

### 4.3 法向场 → 曲面初值

XY 固定，优化顶点高度。目标出射方向为 d_target，Snell 要求的法向满足 N∥n_lens*(0,0,1)-n_air*d_target，因此 ∇h=-Nxy/Nz。法向场一般不完全可积，用 SciPy 稀疏 LSMR 求最小二乘高度初值，再逐级加密、插值旧高度。

### 4.4 JAX 真实折射优化

面片法向来自实际顶点几何。令 η=n_lens/n_air、q=1-η²(1-Nz²)，出射方向 d=η*(0,0,1)+(sqrt(q)-η*Nz)*N。q<0 表示全反射，优化中有障碍惩罚，最终追迹另行统计。

屏幕交点严格是 (X,Y)=(x,y)+(D-h)/dz*(dx,dy)。改变透镜厚度不会移动屏幕。

先使用 Optax Adam 热启动，再用 SciPy L-BFGS-B；JAX jit(value_and_grad) 给出实际函数值和梯度。损失含真实落点误差、法向一致性、轻微曲面正则、全反射障碍及厚度边界；图像阶段加入灰度、图像梯度、暗区和细线加权损失。

### 4.5 局部加密与光通量

细线掩码只增加损失权重，不修改输入图。选取预测落在细线附近的源面片，插入共享边中点，在凸域内重新剖分、传递高度并继续优化。外轮廓保持不变。

每面片光通量等于 XY 投影面积占比。普通面片用基础样本，细线附近可以更多；增加样本时相应降低每条样本权重，保证面片总光通量不变。不能因为多采样就人为提高细线亮度。

### 4.6 独立正向验收

`verify.py` 不把 OT 映射当作最终图。重新在真实凸口径中均匀采样独立 Sobol 光线，定位实际三角面片，算交点高度和几何法向，以 NumPy Snell 定律折射到固定屏幕，生成原始目标分辨率的直方图。

每条光线携带固定通量，全反射、出屏幕、口径缺失及接收光分别统计，不把出屏后的图重新归一化。另存加入入口/出口非偏振 Fresnel 损失的图。

各优化阶段分别追迹并保留结果，优先选择通过数值门槛的最低 RMSE 模型；如果都没通过，则选最低 L1 候选，并明确 optical_quality_pass=false。流程跑通不等于成像达标。

### 4.7 CNC 实体

自由曲面、z=0 平底及凸轮廓侧壁组成实体。Trimesh 检查封闭性、法向一致性、正体积、厚度，并在二进制 STL 重读后检查。lens.stl 是 float32 二进制；lens_precision.stl 保留 float64 文本坐标但较大；另输出 OBJ。STL 本身不记录单位，CAM 中按 mm。

这是无底切的高度场实体；具体刀具半径补偿、步距、抛光和实物加工尚未验证，cnc_release=false。

## 5. 终端与结果保存

终端和 run.log 同时记录 PID、累计耗时、当前阶段、网格数量、SDOT 残差、优化损失和追迹百分比。

不创建 UI，不启动独立显示进程，不调用 plt.show()。Matplotlib 只用 Agg 后端保存图片。--out 指定的文件夹根目录直接放最终选中的模型和图片，无需到阶段子目录中寻找：

```text
my_lens/
  lens.stl                 常规二进制模型
  lens_precision.stl       双精度文本坐标模型
  lens.obj
  mesh.png                 三维网格，颜色映射真实高度
  overview.png             网格、目标、独立模拟总览
  height.png               高度俯视图
  target.png / simulated.png / error.png
  comparison.png / loss.png / loss.csv / mapping.png
  metrics.json / mesh.json
  settings.json / config.json / workflow.json
  result.json / status.json / run.log
  base/ ...                各阶段模型、检查点与独立验收记录
```

若为了观察放大了 z 的显示比例，mesh.png 会标注；数值高度和导出模型不变。最终模型与图像由独立验收选出后复制到根目录，原始阶段数据仍保留。

## 6. 输出

根目录直接包含最终 STL/OBJ、mesh.png、simulated.png、error.png 和 settings.json、config.json、environment.json、input.json、domain.json/npz、processed_target.png、mask_binary.png、run.log、transport.json/npz、candidates.json、result.json、status.json、overview.png。

base/ 和各后续阶段中保存 surface_*.npz、raytrace.npz、metrics.json、mesh.json、loss.csv/png、target.png、simulated.png、error.png、comparison.png、height.png、mapping.png 和三个模型。result.json 指向真正选中的阶段。科学评价使用 NPZ 原始浮点图；PNG 以同一目标峰值显示，超出动态范围会截白。

只提供终端入口；不提供 Python API，不创建心跳线程。计算完成后直接退出；网格高度图固定使用 viridis 色图。

## 7. 实测范围

12 项测试通过，包括 Snell、全反射、通量、图像方向、整数居中、梯度、STL、L 像素映射、凸域采样、斜边退化清理和非正方形/空掩模拒绝。

通用入口已用 n=1.45、L=40 mm、D=120 mm、grid=32 和非圆凸掩模完成多分辨率、局部加密、262,144 条独立光线及模型导出，位于 results/generic_example（最终只保存版本）；该快速运行只证明参数和流程正确衔接，成像质量未通过高质量门槛。原最优模型保持在 results/saturn_centered_poly，历史质量不自动适用于新的参数/掩模。

实现沿用 2014 两阶段思路，固定 XY 高度场、重心插值再真实图像优化，并非论文完整自由 XYZ 重建。几何光学不包含衍射、粗糙度和实际加工误差。

主要模块：settings.py 参数；domain.py 凸掩模与网格；transport.py SDOT；optimize.py JAX；adaptive.py 加密；verify.py 独立追迹；export.py 实体；visualize.py Matplotlib；general_pipeline.py 流程。