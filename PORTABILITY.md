# CausticPy 可迁移性审计

## 已消除的机器绑定

| 原位置 | 处理 |
|---|---|
| run_OT.py 强制 CPU、float64、线程、XLA_FLAGS | 移到可选 runtime；双精度默认 true，其他项省略不改现有环境 |
| run_OT.py 固定 MSYS2 GCC、前置本机 Python 路径 | 源码移除；compiler/path_dirs 由 JSON 指定，或用系统默认 |
| backend.py 再次强制 CPU、float64 | 移除；backend 只导入 JAX 并报告实际环境 |
| backend.py 固定 .cache 和 Windows DLL 目录 | 项目缓存配置已删除；缓存由库管理，dll_dirs 仅保留显式外部目录 |
| activate.ps1 固定用户名、Mamba 安装位置和环境路径 | 移除；环境由用户在终端选择 |
| 输出必须在 CausticPy 内 | 移除；允许任何可写新目录，相对路径规则保持一致 |
| verify.py 未配置仍强制使用全部 CPU | 未指定 CAUSTIC_THREADS 时采用 ThreadPoolExecutor 默认，显式 threads 时仍按配置 |
| 遗留 NPZ 可视化入口 | 移除旧读取器，两个入口各自由 report.py 输出四图 |

common/runtime.py 使用 os.pathsep 和 pathlib，没有按操作系统名称选择配置的分支。runtime 全部为直接字段：提供就严格执行；双精度默认 true，其他项不提供就保留系统设置。dll_dirs 非空时要求当前 Python 提供 DLL 目录注册能力，否则明确报错。当前设备、精度、编译器与缓存环境继续记录在唯一 result.json 中。

OT 的四个 JSON 的 runtime 直接保留本机 GCC 路径，这是显式用户配置，不是 Python 内置回退。任意系统都会严格按提供值执行；迁移时修改/删除不适用的 compiler、path_dirs、dll_dirs，不再自动忽略。源码不扫描开发者软件目录，也不自动激活某个 Python。

## 仍有固定值，属于算法或输出规则

这些值不依赖操作系统，未在可迁移性修改中擅自改变算法：

| 位置 | 固定值 | 含义 |
|---|---|---|
| adaptive.py | 灰度 0.05、骨架距离 2 像素、落点邻域 6 像素 | 当前细线检测与局部加密启发式 |
| optimize.py | 相对灰度 0.05、骨架距离 2、额外采样邻域 4 像素 | 图像损失细线区域与采样增强 |
| optimize.py | 基础采样 4/8/16、图像宽度 512、面片数 150000 | 选择每面片基础采样数量 |
| optimize.py | 初始高度内缩 0.1 mm | 初值避免恰好落在厚度边界 |
| domain.py | 内部网格与边界距离 0.35 倍间距 | 避免过薄边界面片 |
| geometry.py/domain.py | 退化判别 1e-10、面积检查容差、最多 8 轮清理 | 几何数值稳定性 |
| optics.py/verify.py | sqrt/除法中的小正数 | 数值保护，非机器路径或运行环境 |
| verify.py | 每批 2^18 条光线 | 内存和追迹批次组织 |
| target.py/domain.py/transport.py | NumPy float64、SDOT FP64 | 当前数值模型的双精度数据；跨平台一致，不用于强制 JAX 全局精度 |
| optimize.py | 每 25 次目标评价打印 | 日志频率 |
| report.py | viridis、四个固定图片文件名和 DPI | 按用户要求固定输出表现 |
| run_OT.py | 无全反射且最低独立 RMSE | 用户要求的选择准则；未恢复其他质量阈值 |

RGB 灰度系数、像素坐标翻转、+z 平行光、平面入口 z=0 和 Snell 公式中的数学常量同样保留。物理参数、优化权重、求解器选项均已由单个 JSON 提供。内部 Config 仍保留历史默认值，但示例 JSON 显式给出当前任务参数。

## 历史验证范围

- 无 runtime 时仅默认开启 JAX 双精度，其他环境逐项不变，配置函数不创建缓存目录。
- 部分 runtime 只更改已提供项，不覆盖已有 CXX/XLA_FLAGS。
- 真实 JAX 导入分别尊重已有 JAX_ENABLE_X64=false/true。
- 不同系统名下解析相同直接配置；不支持 DLL 注册能力时显式失败，旧的平台分段被拒绝。
- Windows 小规模完整流程与 RUN 指标一致、四图逐像素一致、STL 与原高精度 STL 逐字节一致。
- 真实输出放在 CausticPy 外部仍成功；写入审计只允许每级六文件，无创建后删除。
- 本机 WSL 查询返回 E_ACCESSDENIED，Linux 全流程未实跑。Python 3.14 和各平台可用依赖、C++ 编译器/Ninja 是环境前提。

编译产物绑定操作系统和编译器。迁移时不要复制 .cache；目标系统从源码重新构建。历史 results 中的输入绝对路径是运行记录，不是启动依赖。

缓存字段已从 JSON 和运行设置代码移除。Loom 负责默认缓存及生成 DLL 依赖目录，Matplotlib 使用自身缓存；程序不再维护项目缓存路径。

当前重构中的 OT 快测使用项目已有 Loom 缓存。若在同样的 Windows 环境中需要复用 RUN2 的缓存, 可在终端设置 LOOM_CACHE_DIR 指向 RUN2/.cache/loom, LOOM_BUILD_DIR 指向 RUN2/.cache/build, 并在该次 OT JSON 的 runtime.dll_dirs 中加入现有 build/runtime 和 build/units 绝对路径。
