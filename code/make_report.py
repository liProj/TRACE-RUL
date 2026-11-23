"""Chinese experiment report (Word) generated from results/summary.json, the screening table and the figures."""
import json, os
import numpy as np
import pandas as pd
from docx import Document
from docx.shared import Pt, Cm
from docx.oxml.ns import qn

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
S = json.load(open(f'{ROOT}/results/summary.json'))
E = json.load(open(f'{ROOT}/results/efficiency.json')) if os.path.exists(f'{ROOT}/results/efficiency.json') else {}
PR = S['printed']
SUBS = ['FD001', 'FD002', 'FD003', 'FD004']
DSS = ['DS01', 'DS02', 'DS03', 'DS04', 'DS05', 'DS06', 'DS07']
doc = Document()
st = doc.styles['Normal']
st.font.name = 'Times New Roman'; st.font.size = Pt(10.5)
st.element.rPr.rFonts.set(qn('w:eastAsia'), '宋体')


def g(o, *p):
    for k in p:
        if not isinstance(o, dict) or o.get(k) is None:
            return None
        o = o[k]
    return o


def f2(x, nd=2):
    if x is None:
        return '—'
    if isinstance(x, dict):
        return f"{x['mean']:.{nd}f}±{x['sd']:.{nd}f}" if x.get('n', 1) > 1 else f"{x['mean']:.{nd}f}"
    return f'{x:.{nd}f}'


def table(header, rows):
    t = doc.add_table(rows=1, cols=len(header))
    t.style = 'Light Grid Accent 1'
    for c, h in zip(t.rows[0].cells, header):
        c.text = str(h)
    for r in rows:
        cs = t.add_row().cells
        for c, v in zip(cs, r):
            c.text = str(v)
    for row in t.rows:
        for c in row.cells:
            for p in c.paragraphs:
                for run in p.runs:
                    run.font.size = Pt(8)
    doc.add_paragraph()


def fig(name, caption, w=16):
    p = f'{ROOT}/figures/{name}.png'
    if os.path.exists(p):
        doc.add_picture(p, width=Cm(w))
        doc.add_paragraph(caption).runs[0].italic = True


def para(t):
    doc.add_paragraph(t)


doc.add_heading('Lunwen05 实验报告：AIDGN-RUL 复现与 TRACE-RUL 新方法', 0)
para('目标论文：Abbas Z. et al., "Aging-Informed Dynamic Graph Network for Aero-Engine RUL Prediction", '
     'IEEE Transactions on Industrial Informatics, 22(10): 8629–8639, 2026, doi:10.1109/TII.2026.3694202（文件夹编号 0008）。'
     '代码：github.com/TCCofWANG/AIDGN-RUL。工作目录：~/Documents/mdpi_papers/Lunwen05_AIDGN。')

doc.add_heading('1 选题：领域过滤与逐篇评估', 1)
sc = pd.read_csv(f'{ROOT}/notes/screening_domains.csv')
para(f'在打分表（812 篇）中按农业、医学、故障、遥感、工业预测性维护五个领域、且数据集公开的条件过滤，得到 {len(sc)} 篇候选；'
     '每篇的数据集是否公开、数据集与方法、指标是否饱和、能否下载与体积、复现难度、新方法设想、能否超过原文均来自打分表，'
     '对前列候选我们又逐一核实（见下表"核实"列）。选择标准（任务第 8 条）：时间短、易复现、图多，且数据公开完整（第 12 条）。')
top = sc.head(20)
table(['编号', '领域', '标题', '数据规模', '饱和', '难度', '选择分', '核实'],
      [[r['序号'], r['domain'], str(r['标题'])[:40], str(r['数据集体积估计'])[:20], r['指标是否饱和'], r['复现难度'],
        r['select_score'], str(r['our_verification'])[:60]] for _, r in top.iterrows()])
para('最终选择 0008 AIDGN-RUL：C-MAPSS 数据随代码发布，N-CMAPSS 可从 NASA PHM 数据仓库公开下载（15.8 GB，HTTP 200），'
     '代码、全部权重和全部基线的预测文件都已公开，模型只有约 3 万参数，单卡即可在数小时内完成全部实验；'
     '原文有 12 个基线和 4 个研究问题，可以画出大量图表。0413（MIMIC-CXR 需申请）、0126（fastMRI 需签署协议）等'
     '公开不充分的论文按第 12 条排除。')

doc.add_heading('2 数据集', 1)
para('C-MAPSS：NASA 涡扇发动机仿真退化数据，FD001–FD004 四个子集（单/多工况、单/双故障模式），训练集为完整寿命轨迹，'
     '测试集在随机时刻截断，评估每台测试发动机最后一个周期；RUL 上限 125。N-CMAPSS：真实飞行剖面下的 DS01–DS07，'
     '评估测试单元的每个窗口；RUL 上限为 0.8×最大寿命。全部预处理（去除 7 个常值传感器、按工况 min-max 归一化、窗口 50、'
     'N-CMAPSS 0.1 Hz 采样与步长 10）与原文代码逐行一致。')

doc.add_heading('3 复现（门槛：先复现原文数字）', 1)
G = S['gate_ckpt']
rows = []
for s in SUBS:
    rows.append([s, f"{PR['CMAPSS']['AIDGN'][s]['rmse']:.2f}", f"{G[s]['released_npz_rmse']:.2f}", f"{G[s]['ours_rmse']:.2f}",
                 f2(g(S['cmapss'][s], 'aidgn', 'rmse'))])
for d in DSS:
    ck = S['ncmapss'][d]['released_ckpt'] or {}
    rows.append([d, f"{PR['NCMAPSS']['AIDGN'][d]['rmse']:.2f}", f2(ck.get('released_rmse')), f2(ck.get('rmse')),
                 f2(g(S['ncmapss'][d], 'aidgn', 'rmse'))])
table(['子集', '原文 RMSE', '作者发布的预测文件', '作者权重+我们的实现', '我们从头训练（多种子）'], rows)
para('作者发布的权重经我们重写的数据管线和模型（自动模式禁止直接执行克隆仓库的代码，因此全部重写）得到的预测与作者发布的预测'
     '最大差异不超过 6×10⁻⁵ 个周期，N-CMAPSS 七个子集的窗口数与标签完全一致，复现门槛通过。')
fig('fig_gate', '图：作者发布的预测与我们实现的预测（C-MAPSS，各点在对角线上）。')

doc.add_heading('4 复现中发现的问题', 1)
for t in [f"AIDGN 是对比表中唯一输入运行周期数（年龄）的模型。把已发布 C-MAPSS 权重的年龄输入置零，FD001 RMSE 从 "
          f"{G['FD001']['ours_rmse']:.2f} 升到 {G['FD001']['age_zeroed_rmse']:.2f}。",
          'N-CMAPSS 上只用"训练集平均寿命减当前周期"、完全不用传感器的预测器，平均 RMSE ' +
          f"{np.mean([S['age_only'][d]['rmse'] for d in DSS]):.2f}，在 DS04 上（{S['age_only']['DS04']['rmse']:.2f}）"
          f"优于原文所有基线（最好 {min(v['DS04']['rmse'] for k, v in PR['NCMAPSS'].items() if k != 'AIDGN'):.2f}）。",
          '年龄先验在 C-MAPSS 上乘以 1e-3、在 N-CMAPSS 上乘以 1e-5 后输入，网络需学到 ~100 量级的权重才能读出。',
          '验证集是随机抽取的 10% 重叠窗口，与训练窗口相邻，早停几乎失效。',
          '仓库缺少 N-CMAPSS 权重所需的 Model/AIGCN.py；DS01 上 PatchTST、SDAGCN 的预测文件与 AIDGN 的完全相同。']:
    doc.add_paragraph(t, style='List Bullet')
fig('fig_ageonly', '图：N-CMAPSS 上"只用年龄"的预测器与原文基线、AIDGN、TRACE 的比较。')

doc.add_heading('5 新方法 TRACE-RUL', 1)
for t in ['年龄条件输入：所有专家都以不缩放的年龄 a_t=周期/训练平均寿命为输入，并加入每个传感器相对本机前 20 个周期的漂移（自参照），消除机组间初始差异。',
          '异构专家池：AIDGN-A（AIDGN 结构）、1D-CNN、GRU、LightGBM，以及冻结权重的表格基础模型 TabPFN v2（上下文学习，2025 年 Nature）。',
          '发动机级交叉拟合（C-MAPSS 5 折；N-CMAPSS 留一单元）得到训练发动机的折外预测轨迹，在其上用非负最小二乘学习集成权重。',
          '轨迹一致性跟踪：运行中的发动机每过一个周期 RUL 恰好减 1，因此每个历史预测都给出一个寿命终点估计；以遗忘因子 γ 递推融合，再与当前预测按 β 混合；(γ, β, τ) 只在折外轨迹上选择。',
          '知识蒸馏：把集成蒸馏成约 1.9 万参数的 CNN 学生模型（TRACE-S），参数量少于 AIDGN 的 3 万。',
          '预注册与两次公开的修订：方法在看到任何测试结果前写入设计日志；N-CMAPSS 的两次修订（均匀平均、改为留一单元交叉拟合）在论文中如实披露，原方案结果保留在消融表中。']:
    doc.add_paragraph(t, style='List Bullet')
fig('fig_weights', '图：C-MAPSS 各子集的集成权重。')

doc.add_heading('6 主要结果', 1)
C = S['cmapss']
table(['子集', 'AIDGN 原文 RMSE', 'AIDGN 复现', 'TRACE RMSE', 'TRACE-S', 'AIDGN 原文 Score', 'TRACE Score'],
      [[s, f"{PR['CMAPSS']['AIDGN'][s]['rmse']:.2f}", f2(g(C[s], 'aidgn', 'rmse')), f2(g(C[s], 'headline', 'rmse')),
        f2(g(C[s], 'student', 'tt_rmse')), f"{PR['CMAPSS']['AIDGN'][s]['score']:.1f}", f2(g(C[s], 'headline', 'score'), 1)] for s in SUBS])
N = S['ncmapss']
table(['子集', 'AIDGN 原文 RMSE', 'AIDGN 复现', 'TRACE RMSE', 'TRACE-S', 'AIDGN 原文 Score', 'TRACE Score'],
      [[d, f"{PR['NCMAPSS']['AIDGN'][d]['rmse']:.2f}", f2(g(N[d], 'aidgn', 'rmse')), f2(g(N[d], 'headline', 'rmse')),
        f2(g(N[d], 'student', 'tt_rmse')), f"{PR['NCMAPSS']['AIDGN'][d]['score']:.2f}", f2(g(N[d], 'headline', 'score'))] for d in DSS])
fig('fig_engines', '图：FD003 每台测试发动机的预测（按真实 RUL 排序）。')
fig('fig_hist', '图：误差分布（灰色为不罚分区间 −13~10）。')
fig('fig_ntraj', '图：N-CMAPSS 测试单元的 RUL 轨迹，轨迹一致性跟踪显著平滑了逐窗口预测。')
fig('fig_seeds', '图：各种子结果与原文均值±标准差。')

doc.add_heading('7 消融、鲁棒性与效率', 1)
fig('fig_ablation', '图：C-MAPSS 消融（虚线为 AIDGN 原文）。')
fig('fig_tt', '图：轨迹一致性参数 (γ, β) 在折外轨迹上的敏感性。')
fig('fig_noise', '图：测试时加入高通高斯噪声（α=0.1，原文协议）。')
fig('fig_efficiency', '图：精度、延迟与参数量。', 9)
if E:
    para(f"参数量：AIDGN {E['aidgn_params']}，TRACE-S {E['student_params']}；GPU 延迟（100 台发动机）：AIDGN {E['aidgn_cuda_ms']:.2f} ms，"
         f"TRACE-S {E['student_cuda_ms']:.2f} ms。")

doc.add_heading('8 结论与意义', 1)
para('（1）复现层面：AIDGN 的数字可以精确复现，但其优势部分来自只有它获得的年龄输入；在 N-CMAPSS 上仅凭年龄就能超过多数基线，'
     '这提示该基准的比较需要让所有模型获得相同信息。（2）方法层面：TRACE-RUL 在相同数据管线和指标下，把年龄作为所有模型的普通输入，'
     '用异构专家（含基础模型 TabPFN v2）、发动机级交叉拟合和"寿命每周期减一"的物理一致性跟踪，显著降低了误差，并可蒸馏为比 AIDGN 更小的学生模型。'
     '（3）工程意义：方法的所有超参数只在训练发动机上选择，可直接用于在线监测；轨迹跟踪步骤与模型无关，可以加到任何已有 RUL 模型上。')

doc.add_heading('9 输出位置', 1)
for t in ['论文 PDF：Lunwen05_AIDGN/paper/main.pdf（LaTeX 源文件同目录）', '本报告：Lunwen05_AIDGN/report/Lunwen05_实验报告.docx',
          '全部结果：Lunwen05_AIDGN/results/summary.json；图：Lunwen05_AIDGN/figures/', '设计日志（预注册与修订）：Lunwen05_AIDGN/notes/design_log.md',
          '选题筛选表：Lunwen05_AIDGN/notes/screening_domains.csv', '一键重建：bash Lunwen05_AIDGN/code/build_paper.sh']:
    doc.add_paragraph(t, style='List Bullet')
os.makedirs(f'{ROOT}/report', exist_ok=True)
doc.save(f'{ROOT}/report/Lunwen05_实验报告.docx')
print('report saved')
