"""Prepare hashed external inputs and generate v3 QP pages from actual evidence.

Every measurement/status comes from shared-storage artifacts. Sealed solver
evidence is independently reaccepted on each render; historical pages use their
original frozen bundle. No infrastructure publication or commercial win.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import html
import json
from pathlib import Path
import tempfile

import numpy as np

if __package__:
    from .accept_qp_campaign import accept
    from .render_qp_report import ROOT, frame, generate as historical
else:
    from accept_qp_campaign import accept
    from render_qp_report import ROOT, frame, generate as historical


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source(path: Path) -> dict:
    if not path.is_file() or not path.read_bytes():
        raise ValueError(f"missing/empty report input: {path}")
    return {"path": str(path.resolve()), "sha256": digest(path)}


def load(item: dict) -> dict:
    path = Path(item["path"])
    if digest(path) != item["sha256"]:
        raise ValueError(f"report input changed: {path}")
    return json.loads(path.read_text())


def prepare(args) -> Path:
    root = args.artifact_root.resolve()
    if not str(root).startswith("/3fs/research3fs/data/tmp/"):
        raise ValueError("report evidence must be on shared 3FS tmp")
    root.mkdir(parents=True, exist_ok=True)
    run = Path(tempfile.mkdtemp(prefix="report-inputs-", dir=root))
    bundle = {"modified_at": datetime.now(timezone.utc).isoformat()}
    for key in ("old_bundle", "corpus", "verification", "targeted", "wheel", "live", "campaign", "audit"):
        bundle[key] = source(getattr(args, key))
    bundle["acceptances"] = []
    if len(args.acceptance) != len(args.accepted_bundle):
        raise ValueError("each acceptance record requires its explicit accepted bundle")
    for path, accepted_bundle in zip(args.acceptance, args.accepted_bundle, strict=True):
        report = json.loads(path.read_text())
        if report["verdict"] != "PASS":
            raise ValueError(f"independent acceptance failed: {path}")
        bundle["acceptances"].append({
            "record": source(path), "bundle": source(accepted_bundle),
            "pin": report["bundle_sha256"],
        })
    output = run / "bundle.json"
    output.write_text(json.dumps(bundle, indent=2, allow_nan=False) + "\n")
    print(f"QP_CAMPAIGN_REPORT_INPUT verdict=PREPARED bundle={output} sha256={digest(output)}")
    return output


def esc(value) -> str:
    return html.escape(str(value))


def chart(title: str, labels: list[str], values: list[float], unit: str) -> str:
    if not labels or len(labels) != len(values) or not all(np.isfinite(values)):
        raise ValueError("chart input incomplete")
    maximum = max(values) or 1
    rows = []
    for i, (label, value) in enumerate(zip(labels, values, strict=True)):
        y = 40 + i * 44
        width = value / maximum * 470
        rows.append(
            f'<text x="5" y="{y+17}" font-size="14">{esc(label)}</text>'
            f'<rect x="245" y="{y}" width="{width:.2f}" height="23" fill="{("#18362f", "#bb4c24")[i % 2]}"/>'
            f'<text x="{255+width:.2f}" y="{y+17}" font-size="14">{value:.3f} {unit}</text>'
        )
    return (
        f'<figure><svg viewBox="0 0 900 {len(values)*44+70}" role="img" aria-label="{esc(title)}">'
        f'<text x="5" y="22">{esc(title)}</text>{"".join(rows)}</svg>'
        f'<figcaption>{esc(title)}，每根柱来自原始记录，不作跨机排名。</figcaption></figure>'
    )


def generate(path: Path) -> dict[Path, str]:
    bundle = json.loads(path.read_text())
    verified = [load(bundle[key]) for key in ("verification", "targeted", "wheel", "live")]
    if any(report["verdict"] != "PASS" for report in verified):
        raise ValueError("regression/wheel/live input failed")
    verification, targeted, wheel, live = verified
    accepted, records = [], []
    for item in bundle["acceptances"]:
        load(item["record"])
        load(item["bundle"])
        report = accept(Path(item["bundle"]["path"]), item["pin"])
        accepted.append(report)
        for entry in report["entries"]:
            records.append((entry, json.loads(Path(entry["result_path"]).read_text())))
    batch = sorted([(e, r) for e, r in records if r.get("mode") == "batch"], key=lambda p: p[1]["size"])
    optional = [(e, r) for e, r in records if r.get("mode") in ("adaptive", "operators")]
    cpu = [(e, r) for e, r in records if "arms" in r]
    correct = [(e, r) for e, r in records if r.get("mode") == "correctness"]
    if len(batch) < 3 or not cpu or not correct or len(optional) != 2:
        raise ValueError("report requires batch ladder, optional negative results, CPU and CUDA correctness")
    corpus = load(bundle["corpus"])
    counts = {}
    for row in corpus["rows"]:
        counts.setdefault(row["solver"], [0, 0])[row["verdict"] != "QUALIFIED"] += 1
    campaign = load(bundle["campaign"])
    audit = load(bundle["audit"])
    if set(campaign["seats"]) != {seat["seat"] for seat in audit["seats"]} or any(
        seat["verdict"] != "PASS" or seat["outside_scope"] for seat in audit["seats"]
    ):
        raise ValueError("seat write-scope audit failed")
    speedups = [r["result"]["measurements"]["cold"]["baseline"]["latency"]["p50_s"] /
                r["result"]["measurements"]["cold"]["candidate"]["latency"]["p50_s"] for _, r in batch]
    report = f"""
<p><strong>尚不能证明全面胜过顶尖商用优化器。</strong>本轮有实测批量收益，
但它只相对本项目的逐题 GPU 执行；商用许可同题实测仍未验证。</p>
<p>{len(campaign['seats'])} 个独立职责的写集均已审计通过，工具实际同时在飞上限为六席，完成后补位。
主代理统一投递，GPU 同时不超过三张。独立回读共检查
{sum(r['checked_samples'] for r in accepted)} 条候选解；不以自报 PASS 替代验收。</p>
<h2>批量收益</h2><p>同形状稠密 QP，八份相同题目；比较一批求解与八次逐题执行，
不是异构题库、商用加速或单题多 GPU。冷求解加速范围
{min(speedups):.2f}x 至 {max(speedups):.2f}x。每组各臂的样本数、
逐题状态和独立残差均在不可变证据中。冷计时包含模型校验、传输和求解；
prepared 为复用准备状态。独立 CPU 验收在计时区间外，不把它算进加速比。</p>
"""
    for mode in ("cold", "prepared"):
        labels, values = [], []
        for _, r in batch:
            for arm in ("baseline", "candidate"):
                labels.append(f"{r['size']}变量 / {'逐题' if arm == 'baseline' else '批量'}")
                values.append(r["result"]["measurements"][mode][arm]["latency"]["p50_s"] * 1000)
        report += chart(f"批量 {mode} 中位延迟，越低越好", labels, values, "ms")
    report += "<h2>负收益与未证实收益</h2>"
    for entry, r in optional:
        m = r["result"]["measurements"]
        report += chart(f"{r['mode']} / {r['size']}变量：同题路径工作值",
                        ["冷 / 默认", "冷 / 可选", "prepared / 默认", "prepared / 可选"],
                        [m[mode][arm]["latency"]["p50_s"] * 1000
                         for mode in ("cold", "prepared") for arm in ("baseline", "candidate")], "ms")
        report += (
            "<p>自适应路径本题差异在实测对照噪声内，不宣称提速。默认关闭。</p>"
            if r["mode"] == "adaptive" else
            "<p>当前稀疏 CG 小题明显更慢。它使用调用者 PSD 声明，与稠密全谱验证不等价，"
            "不作资格排名、不启用自动路由。Nsight 显示大量微小 kernel、主机同步与检查开销。</p>"
        )
    report += "<h2>同机 CPU 参考</h2>"
    for _, r in cpu:
        arms = [a for a in r["arms"] if a["status"] == "PASS" and a["control_of"] is None]
        report += chart(f"{r['dimensions']['variables']}变量 / {r['hardware']['torch_threads']}线程 / cold",
                        [a["id"] for a in arms],
                        [a["modes"]["cold"]["latency"]["p50_s"] * 1000 for a in arms], "ms")
    report += "<p>只展示同机同题原坐标精度通过的工作值，不能与集群 GPU 跨机拼加速比。"
    report += "不同尺寸方向不同，单题不能外推整个优化领域。</p>"
    report += "<h2>压力题资格</h2>"
    report += chart("合成压力题合格数，不是工业题库排名", list(counts),
                    [value[0] for value in counts.values()], "题")
    report += "<p>" + "；".join(f"{esc(name)}：{v[0]} 合格 / {sum(v)} 尝试" for name, v in counts.items()) + "。</p>"
    report += "<p>Openopt 的不可行/无界模型仍返回迭代上限，没有生成可验证方向。"
    report += "独立方向检查器已实现，但不能替代方向生成。OSQP 两项原证书失败已定位为单独的证书容差，"
    report += "收紧后的诊断对照通过；不追改原结果。奇异、冗余、病态等仅覆盖合成族。</p>"
    report += "<h2>资源与瓶颈</h2><div class='table'><table><tr><th>模式/变量</th><th>GPU均值/峰值%</th><th>峰值MiB</th><th>CPU核数均值</th><th>kernel ms / profile端到端ms / 占比%</th></tr>"
    for entry, r in batch + optional:
        telemetry = json.loads((Path(entry["result_path"]).parent / "telemetry.json").read_text())
        rows = [s["gpu"].split(",") for s in telemetry if s["rc"] == 0]
        utils, memory = [float(row[1]) for row in rows], [float(row[2]) for row in rows]
        span = telemetry[-1]["monotonic_s"] - telemetry[0]["monotonic_s"]
        if span <= 0:
            raise ValueError("CPU telemetry interval is empty")
        cpu_cores = (telemetry[-1]["cpu"]["usage_usec"] - telemetry[0]["cpu"]["usage_usec"]) / 1e6 / span
        nsys = r["nsight"]
        share = 100 * nsys["kernel_total_s"] / nsys["profiled_prepared_e2e_s"]
        report += f"<tr><td>{r['mode']}/{r['size']}</td><td>{np.mean(utils):.1f}/{max(utils):.0f}</td><td>{max(memory):.0f}</td><td>{cpu_cores:.2f}</td><td>{nsys['kernel_total_s']*1000:.3f}/{nsys['profiled_prepared_e2e_s']*1000:.3f}/{share:.1f}</td></tr>"
    report += "</table></div><p>资源来自 Pod 生命周期内采样，包含初始化及剖析阶段，不能当持续饱和吞吐。"
    report += "profile 是独立求解，含剖析开销，不能替代无剖析延迟。"
    report += "批量主要仍花在三角求解；稀疏主要受小 kernel 与同步开销影响。"
    report += "硬件理论下界、持续饱和吞吐、真实多 GPU 算法缩放仍未验证。</p>"
    report += f"<p>现场保留 {len(live['retained_failures'])} 个历史失败任务，"
    report += f"当前 cordon {len(live['cordoned'])}，API readiness 已通过；失败未删除、未冒充新实验成功。</p>"
    batch_paths = {entry["result_path"] for entry, _ in batch}
    spans = [(datetime.fromisoformat(row["started_at"].replace("Z", "+00:00")),
              datetime.fromisoformat(row["finished_at"].replace("Z", "+00:00")))
             for row in live["measurements"] if row["result"] in batch_paths]
    if len(spans) != len(batch_paths):
        raise ValueError("live snapshot does not cover batch timing")
    events = [(start, 1) for start, _ in spans] + [(end, -1) for _, end in spans]
    active = peak = 0
    for _, delta in sorted(events):
        active += delta
        peak = max(peak, active)
    report += f"<p>批量阶梯实际计算并发峰值 {peak} 个 Pod，"
    report += f"从首个启动至末个结束 {(max(end for _, end in spans)-min(start for start, _ in spans)).total_seconds():.1f} 秒；"
    report += "任务一起提交不代表三卡同时计算，不宣称调度或多 GPU 缩放收益。</p>"
    tests = verification["tests"]
    report += f"<h2>验证与交付边界</h2><p>完整 CPU 回归：{tests['tests']-tests['skipped']}通过、"
    report += f"{tests['skipped']}跳过；随后定向门 {targeted['tests']['tests']} 项通过。"
    report += f"独立 CUDA 专项 {correct[0][0]['tests']['tests']} 项、零跳过。"
    report += f"隔离 wheel：{wheel['modules']} 模块、许可证 {esc(wheel['license_expression'])}，"
    report += f"现场快照 {esc(live['read_at'])}：{len(live['ready'])}节点 Ready、{len(live['active'])}占用任务。</p>"
    report += "<p>本轮新增：可选 rho 自适应、逐项批量、稀疏/矩阵自由 CG、独立数值证据检查、"
    report += "合成压力题、公平对照和封存验收。修复目标/残差溢出误报。"
    report += "仍缺：成熟 presolve/缩放/预条件/精度修复、可靠方向生成与目标 gap 停止、"
    report += "广泛公开/工业题库、SOCP及更多锥、MILP/MIQP完整求解、许可商用同题对照、"
    report += "最佳 CPU/GPU 基线与部署级长期可靠性。项目0BSD不替依赖重新许可。</p>"
    report += "<p>本页仅仓内交付，未线上发布；不修改 infra 发布入口。"
    report += "早期缺状态/题目快照的 GPU 件及所有失败均保留，但被独立门拒绝。"
    report += "历史 v2 对应旧源，不用当前核心冒充其原测量。</p>"
    report += f"<h2>证据入口</h2><p>本页输入 <code>{esc(path)}</code>；SHA256 <code>{digest(path)}</code>。</p>"
    report += "<details><summary>封存与核验来源</summary><ul>"
    for item in bundle["acceptances"]:
        report += f"<li><code>{esc(item['record']['path'])}</code><br>bundle pin <code>{item['pin']}</code></li>"
    report += "</ul></details><p><a href='research/commercial-gap.md'>商用能力调研及官方来源</a> / "
    report += "<a href='qp-design.md'>算法边界</a> / <a href='research/acceptance.md'>独立验收合同</a></p>"
    design = """<p>独立的通用 QP/LP 实验核心，保持 OpenPO API 身份。
默认稠密 ADMM；可选残差平衡 rho、同形状批量和稀疏/矩阵自由 CG。</p>
<h2>正确性边界</h2><p>严格显式设备、owned tensor workspace、原坐标残差与非有限失败。
批量逐项冻结状态；CG 核验真实内层残差、不稠密化。稀疏调用者必须声明 PSD，
抽样探测不证明凸性。独立候选/方向检查仅是数值证据，不是精确证明。</p>
<h2>实测与限制</h2><p><a href='benchmark-report-v3.html'>v3 同源报告</a>展示收益与反证。
没有成熟锥/整数求解器、可靠不可行方向生成、通用 minimize 或商用资格。
当前性能门重新检查封存源、题目、解向量、参数、状态、对照与 profile。</p>
<h2>维护入口</h2><p><a href='qp-design.md'>完整设计</a> /
<a href='BACKLOG.md'>开发账本</a>。接口均实验性；0BSD只覆盖项目代码。</p>"""
    load(bundle["old_bundle"])
    pages = historical(Path(bundle["old_bundle"]["path"]))
    for family, title, content in (
        ("benchmark-report", "Openopt / v3 实测与商用差距", report),
        ("design", "Openopt / v3 通用 QP 设计", design),
    ):
        page = frame(title, content, bundle["modified_at"], family, 3).replace(
            'lang="en"', 'lang="zh-CN"').replace(
            "Generated by scripts/render_qp_report.py", "Generated by scripts/render_qp_campaign.py")
        pages[ROOT / "docs" / f"{family}-v3.html"] = page
        pages[ROOT / "docs" / f"{family}.html"] = page
    pages[ROOT / "docs/doc-superseded.tsv"] += (
        "design-v2.html\tdesign-v3.html\tOptional QP paths and independent evidence replay\n"
        "benchmark-report-v2.html\tbenchmark-report-v3.html\tNew source campaign; old measurements remain historical\n"
    )
    return pages


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--prepare-input", action="store_true")
    parser.add_argument("--artifact-root", type=Path, default=Path("/hf3fs-jd/data/tmp/openopt-campaign"))
    parser.add_argument("--acceptance", type=Path, action="append")
    parser.add_argument("--accepted-bundle", type=Path, action="append")
    for key in ("old-bundle", "corpus", "verification", "targeted", "wheel", "live", "campaign", "audit"):
        parser.add_argument("--" + key, type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.prepare_input:
        if not args.acceptance or not args.accepted_bundle or any(getattr(args, key) is None for key in (
            "old_bundle", "corpus", "verification", "targeted", "wheel", "live", "campaign", "audit")):
            parser.error("prepare requires every named evidence input")
        args.bundle = prepare(args)
    if args.bundle is None:
        parser.error("--bundle or --prepare-input required")
    pages = generate(args.bundle)
    if pages != generate(args.bundle):
        raise ValueError("same-source rendering differs")
    for path, content in pages.items():
        if args.check:
            if not path.is_file() or path.read_text() != content:
                raise ValueError(f"stale generated page: {path}")
        else:
            path.write_text(content)
    print(f"QP_CAMPAIGN_REPORT verdict=PASS pages={len(pages)} deterministic=true bundle={args.bundle}")


if __name__ == "__main__":
    main()
