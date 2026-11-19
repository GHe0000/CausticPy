# CausticPy

OT 和 DP 是两个独立的终端程序。运行前准备 Python 3.14 和 requirements.txt 中的依赖。

```text
CausticPy/
├── OT/               SDOT、凸掩模网格、曲面拟合、细化与评估
│   └── config/       四份独立 JSON
├── DP/               几何、曲率初始化、可微面积渲染、损失、L-BFGS 与报告
│   └── config/       default.json
├── common/           NumPy 出射折射与投影、目标图、STL、运行环境
├── examples/         可共用的示例图像
├── run_OT.py
└── run_DP.py
```

在 CausticPy 目录运行以下 PowerShell 命令。已有激活的 caustic-jax 环境时，也可直接从 python 开始运行。

```powershell
mamba run -n caustic-jax python -u .\run_OT.py --config .\OT\config\general_quick.json --out .\OT\results\quick_01
mamba run -n caustic-jax python -u .\run_DP.py --config .\DP\config\default.json --out .\DP\results\dp_01
```

OT 读取目标 PNG 和凸掩模，完成半离散最优输运、曲面优化、独立追迹和导出。默认读取 general_high.json。general_circle_full.json 提供 60 mm 圆形口径、100 mm 屏幕距离和 600 网格设置。后续细化阶段可以用 stages=[] 关闭。候选按独立追迹的相对 RMSE 排序，只选择无全反射的候选。

DP 读取用户指定的初始 STL 和目标 PNG。默认 input_stl 指向 JAX3/RUN 已有的历史模型，可在 DP/config/default.json 中换成自己的 STL。曲率辅助变量由初始曲面拟合，损失公式在 DP/loss.py，权重在 JSON 中。DP 可用 --iterations 覆盖迭代预算。

两个入口各自读取配置、运行和写入结果。common/projection.py 提供前向几何公式，两个算法各自保留训练渲染和独立追迹采样方式。长度单位为 mm，屏幕距离从平面入口 z=0 测量。JSON 相对路径以配置文件目录为基准，命令行相对路径以当前目录为基准。

结果目录必须是新目录。成功的根结果及 OT 的各个候选目录保存 lens.stl、height.png、comparison.png、loss.png、mapping.png 和 result.json。日志带时间戳，直接输出到终端。

runtime 提供可选的设备、精度、线程、编译器和 DLL 路径配置。当前 OT 示例 JSON 明确设置了本机的编译器与 DLL 目录，迁移时可改为目标机路径，或删除这些字段并使用环境中的工具链。Loom 缓存仍由环境或库管理。参见 PORTABILITY.md。
