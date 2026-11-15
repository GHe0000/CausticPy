"""Summarize completed real runs; selection uses independent imaging metrics."""
from pathlib import Path
import json
from audit import audit
root=Path(__file__).resolve().parent
names=['saturn_600','saturn_600_native','saturn_600_thin','saturn_600_adaptive','saturn_600_final','saturn_600_detail_v2','saturn_600_image','saturn_600_fineflux','saturn_centered_v2','saturn_centered_poly']
rows=[]
for name in names:
    folder=root/'results'/name
    if not (folder/'metrics.json').exists() or not (folder/'status.json').exists():continue
    audit(folder)
    m=json.loads((folder/'metrics.json').read_text());mesh=json.loads((folder/'mesh.json').read_text());status=json.loads((folder/'status.json').read_text())
    config=json.loads((folder/'config.json').read_text())
    rows.append(dict(name=name,metrics=m,mesh=mesh,status=status,centered=config.get('center_target',False)))
centered=[r for r in rows if r['centered']]
eligible=[r for r in centered if r['status']['optical_quality_pass']]
best=min(eligible,key=lambda r:r['metrics']['rmse_relative']) if eligible else max(centered or rows,key=lambda r:r['metrics']['thin_flux_ratio'])
selection=dict(result=str(root/'results'/best['name']),quality_pass=bool(eligible),metrics=best['metrics'],mesh=best['mesh'])
(root/'LATEST_RESULT.json').write_text(json.dumps(selection,indent=2),encoding='utf-8')
(root/'results/comparison.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
lines=['# 实际计算结果','',
'全部正式候选均用同一目标、同一 60 mm 口径、相同 Sobol 种子和 16,777,216 条光线独立追迹；表中光通量不作出屏幕重归一化。','',
'| 候选 | 图案居中 | 顶面顶点 | 暗区漏光/接收光 | 相对 RMSE | 细线通量比 | 全部数值门槛通过 |',
'|---|---|---:|---:|---:|---:|---|']
for r in rows:
    m=r['metrics']
    lines.append(f"| {r['name']} | {r['centered']} | {r['mesh']['vertices']//2:,} | {m['dark_leak_captured']:.3%} | {m['rmse_relative']:.4f} | {m['thin_flux_ratio']:.2%} | {r['status']['optical_quality_pass']} |")
lines+=['',f"当前候选：{best['name']}。具体模型与图片位于 LATEST_RESULT.json 的 result 路径。",
'','图案居中=False 为历史保留原始画布位置的对照。最终候选只从图案居中=True 的模型中选择。','','主体门槛：暗区漏光<2%、相对 RMSE<0.35、无 TIR、屏幕接收光通量>98%。细线门槛：原图细线像素通量为目标的 80%–120%。两者都通过才设置 optical_quality_pass。',
'','物理参考：入光平面 z=0，屏幕 z=100 mm；每条出射光线从真实曲面高度传播到固定屏幕。主指标为理想无反射模型；Fresnel 结果另存。',
'','制造：实体是封闭、正体积、没有底切的单值高度场，检查了厚度和 STL 重读。尚未验证具体刀具半径、步距及抛光工艺，cnc_release=False；这不代表已获得 G-code 或实物加工验证。',
'','算法采用 2014 论文的两阶段思想，SDOT 灰度测度/功率图 + JAX Snell 自由曲面优化；采用固定 XY 三角网格和局部加密，源目标重心间为分片线性插值，并非论文完整 XYZ 自适应重建。',
'','验证：8 项物理/网格测试通过；ASCII STL 坐标往返测试通过；不规则网格 CLI 续算回归逐项保留 XY 和面片。串并行追迹相同 Sobol 样本的几何图逐像素一致。','']
(root/'RESULTS.md').write_text('\n'.join(lines),encoding='utf-8')
print(json.dumps(selection,indent=2))
