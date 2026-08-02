"""Generate EXECUTIVE_SUMMARY.pdf: short, investor-friendly, non-technical pitch (EN)."""
import os, math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(BASE, 'tests', 'report_figs')

# ---------------- fig: cost gap ----------------
def fig_cost():
    fig, ax = plt.subplots(figsize=(11, 3.2))
    x = [0, 1]
    labels = ['Datacenter GPU\n(H100-class)\n$25K-40K, 80GB+', 'Consumer GPU\n(RTX 3060)\n~$200, 6GB']
    cost = [30000, 200]
    bars = ax.bar(x, cost, color=['#F59E0B', '#10B981'], width=0.55)
    for b, v in zip(bars, cost):
        ax.text(b.get_x() + b.get_width()/2, v * 1.03, f'${v:,.0f}', ha='center', fontsize=13, fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel('hardware cost (single GPU)', fontsize=11)
    ax.set_ylim(0, 36000)
    ax.text(0.5, 0.90, '~150x hardware cost gap — this is why AI training is reserved\nfor big budgets today. We are turning this gap into a product.',
            transform=ax.transAxes, ha='center', fontsize=11, color='#374151')
    fig.tight_layout(); fig.savefig(os.path.join(FIG, 'exec_cost.png'), dpi=180); plt.close(fig)

fig_cost()
print('figures done')

# ================= PDF =================
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.platypus import (BaseDocTemplate, PageTemplate, Frame, Paragraph, Spacer,
                                Image, Table, TableStyle, PageBreak, HRFlowable)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

dv = os.path.join(os.path.dirname(matplotlib.__file__), 'mpl-data', 'fonts', 'ttf')
pdfmetrics.registerFont(TTFont('DejaVu', os.path.join(dv, 'DejaVuSans.ttf')))
pdfmetrics.registerFont(TTFont('DejaVu-Bold', os.path.join(dv, 'DejaVuSans-Bold.ttf')))
pdfmetrics.registerFont(TTFont('DejaVu-Oblique', os.path.join(dv, 'DejaVuSans-Oblique.ttf')))
F = 'DejaVu'

W, H = A4
M = 2.0 * cm

def page_footer(canvas, doc):
    canvas.saveState()
    canvas.setFont(F, 8)
    canvas.setFillColor(colors.grey)
    canvas.drawString(M, 1.1 * cm, 'Investor Summary — confidential draft, not for distribution before a meeting')
    canvas.drawRightString(W - M, 1.1 * cm, f'page {doc.page}')
    canvas.restoreState()

doc = BaseDocTemplate(os.path.join(BASE, 'EXECUTIVE_SUMMARY.pdf'), pagesize=A4,
                      leftMargin=M, rightMargin=M, topMargin=1.8 * cm, bottomMargin=1.8 * cm,
                      title='Executive Summary — LLM Training on a 6GB GPU', author='V3 Project')
frame = Frame(M, 1.8 * cm, W - 2 * M, H - 3.6 * cm, id='f')
doc.addPageTemplates([PageTemplate(id='page', frames=[frame], onPage=page_footer)])

S = {}
S['h1'] = ParagraphStyle('h1', fontName='DejaVu-Bold', fontSize=19, leading=25, spaceAfter=10, textColor=colors.HexColor('#1E3A8A'))
S['h2'] = ParagraphStyle('h2', fontName='DejaVu-Bold', fontSize=13.5, leading=18, spaceBefore=14, spaceAfter=6, textColor=colors.HexColor('#1E3A8A'))
S['body'] = ParagraphStyle('body', fontName=F, fontSize=10.5, leading=16, alignment=TA_JUSTIFY, spaceAfter=6)
S['box'] = ParagraphStyle('box', fontName=F, fontSize=10.5, leading=16, alignment=TA_JUSTIFY)
S['cap'] = ParagraphStyle('cap', fontName='DejaVu-Oblique', fontSize=8.5, leading=11, textColor=colors.grey,
                          alignment=TA_CENTER, spaceBefore=3, spaceAfter=10)

el = []

# ---- cover ----
el.append(Spacer(1, 0.6 * cm))
el.append(Paragraph('INVESTOR SUMMARY', ParagraphStyle('t', parent=S['h1'], fontSize=24, leading=30, alignment=TA_CENTER)))
el.append(Paragraph('We Fit Large Language Model Training on a 6GB Laptop GPU', ParagraphStyle('s', parent=S['h1'], fontSize=15, leading=21, alignment=TA_CENTER, textColor=colors.HexColor('#F59E0B'))))
el.append(Spacer(1, 0.3 * cm))
el.append(Paragraph('Today, training a 1B+ parameter model requires $25-40K datacenter GPUs and six-figure training budgets. '
                    'Our technology stack makes the same training runnable on a **~$200 consumer GPU (6GB)** — at a scale '
                    'competitors cannot even attempt on the same hardware.', ParagraphStyle('lead', parent=S['body'], fontSize=11.5, leading=18, alignment=TA_CENTER, textColor=colors.HexColor('#4B5563'))))
el.append(Spacer(1, 0.4 * cm))
el.append(Image(os.path.join(FIG, 'exec_cost.png'), width=16.5 * cm, height=16.5 * cm * 0.32))
el.append(Spacer(1, 0.3 * cm))

el.append(HRFlowable(width='100%', thickness=1, color=colors.HexColor('#D1D5DB')))
el.append(Paragraph('THE PROBLEM', S['h2']))
el.append(Paragraph(
    'AI training has become a hardware race: as models grow, memory and compute needs compound and training costs run into '
    'the millions. The result: only big-budget players can participate. Meanwhile, millions of capable consumer GPUs '
    '(6-8GB) sit in the world, written off as "not enough" for AI training.', S['body']))

el.append(Paragraph('OUR PRODUCT — THREE TECHNOLOGIES, ONE GOAL: THE SAME JOB, MUCH CHEAPER', S['h2']))
prows = [['Technology', 'What it does', 'Measured gain'],
         ['V3-Embed\n(vocabulary compression)', 'Compresses the model\'s vocabulary matrix into tiny learned factors, so vocabulary size no longer bloats memory.', '500× compression'],
         ['Analytic Chunked Head\n(output-layer accelerator)', 'Computes training for 50K-250K+ class output layers with a memory-friendly method that is mathematically identical.', '5-12× speedup; ends memory paging'],
         ['Progressive Low-Rank\n(slim hidden layers)', 'Converts part of the hidden layers to low-cost equivalents, fitting wide models into a 6GB budget.', '~3.6× fewer params at same width'],
         ['NCE Sampled Head\n(throughput multiplier)', 'Trains the output layer on target + sampled negatives instead of all classes, so compute cost becomes fully independent of vocabulary size. Hybrid NCE warm-up + exact-CE refine captures both speed and quality.', '3-4.4× extra speed; hybrid loses only ~4% quality']]
pt = Table(prows, colWidths=[4.6 * cm, 8.2 * cm, 3.6 * cm])
pt.setStyle(TableStyle([('FONTNAME', (0, 0), (-1, -1), F), ('FONTSIZE', (0, 0), (-1, -1), 9),
                        ('FONTNAME', (0, 0), (-1, 0), 'DejaVu-Bold'),
                        ('FONTNAME', (0, 0), (0, -1), 'DejaVu-Bold'),
                        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#EFF6FF')),
                        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#D1D5DB')),
                        ('TOPPADDING', (0, 0), (-1, -1), 5), ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
                        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE')]))
el.append(pt)
el.append(Spacer(1, 0.3 * cm))

el.append(Paragraph('EVIDENCE — MEASURED ON THE SAME 6GB GPU', S['h2']))
krows = [['Measurement', 'Vanilla approach (competitor method)', 'Our technology'],
         ['Training speed (same model, same batch)', '3.1K tok/s — memory overflow (9.1GB)', '16.8K tok/s — 5.4x faster, fits in 6GB'],
         ['250K-vocabulary model', 'Cannot even start (output layer alone ~8GB)', 'Trained inside 6GB (~25 min final finetune)'],
         ['WikiText-103 model (50K vocab)', 'Memory-paging at the same load', 'val PPL 95.18 — on 6GB, ~9 GPU-hours'],
         ['229M-param dense model', 'Does not fit 6GB (constant paging)', 'Low-cost equivalent runs in 2.8GB'],
         ['250K-vocab training speed (same model)', '5.0K tok/s (exact CE)', '14.9K tok/s — 3× with NCE'],
         ['8.9M-param model training speed (same model)', '20.6K tok/s', '90.9K tok/s — 4.4×'],
         ['Quality + speed (hybrid NCE warm-up + exact refine)', '224.21 in 14 min (exact CE 2000 steps)', '233.10 in same time — 96% of quality, 3.5× less time than pure NCE']]
kt = Table(krows, colWidths=[4.4 * cm, 5.8 * cm, 6.2 * cm])
kt.setStyle(TableStyle([('FONTNAME', (0, 0), (-1, -1), F), ('FONTSIZE', (0, 0), (-1, -1), 8.5),
                        ('FONTNAME', (0, 0), (-1, 0), 'DejaVu-Bold'),
                        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#EFF6FF')),
                        ('BACKGROUND', (1, 1), (1, -1), colors.HexColor('#FEF2F2')),
                        ('BACKGROUND', (2, 1), (2, -1), colors.HexColor('#ECFDF5')),
                        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#D1D5DB')),
                        ('TOPPADDING', (0, 0), (-1, -1), 4), ('BOTTOMPADDING', (0, 0), (-1, -1), 4)]))
el.append(kt)
el.append(Spacer(1, 0.2 * cm))
el.append(Image(os.path.join(FIG, 'scale.png'), width=16.8 * cm, height=16.8 * cm * 0.36))
el.append(Paragraph('Same hardware, same model body, same batch size: red bar = vanilla (competitor) method; blue/orange bars = '
                    'our technology. Red point: at the scale vanilla cannot even attempt, we train.', S['cap']))

el.append(Paragraph('WHY NOW, WHY US', S['h2']))
for txt in [
    '<b>Market opening:</b> when training costs drop 30-100x, thousands of companies and research teams that say "AI training is too expensive" can enter — with their own data, on their own hardware.',
    '<b>Defensible product:</b> the memory-bound optimizations win exactly at the scale competitors cannot apply on the same hardware (250K vocab, wide models). Four patentable components: V3-Embed, analytic chunked head, progressive low-rank, NCE sampled head.',
    '<b>Chain of evidence:</b> every result was measured on a single 6GB GPU with controlled comparisons; the training accelerator\'s mathematical equivalence was verified bit-by-bit.',
    '<b>Low entry cost:</b> the entire stack was developed on ~$200 hardware — productization targets the same hardware class.',
]:
    el.append(Paragraph('• ' + txt, S['body']))

el.append(Spacer(1, 0.2 * cm))
el.append(HRFlowable(width='100%', thickness=1, color=colors.HexColor('#D1D5DB')))
el.append(Paragraph('HONEST LIMITS', S['h2']))
el.append(Paragraph(
    'In the spirit of transparency: at equal parameter counts and small scale, a classic dense architecture can remain ahead '
    'on quality. The NCE speed layer is an estimator, not the exact distribution — at equal step counts its val PPL trails '
    'exact CE by ~2.8×; its ~4.4× speed lets the same wall-clock budget consume far more data/model, and it scales '
    'independently of vocabulary size. Exact CE remains the metric for quality. '
    'Our win is at the scale competitors **cannot attempt** (wide model + large vocabulary + consumer GPU) — '
    'and that is exactly the region our product targets. We claim a measurable language-model metric (val PPL) with '
    'comparable values; we do not claim "magic" like equal quality at fewer parameters.', S['body']))

el.append(Spacer(1, 0.4 * cm))
el.append(Paragraph('NEXT STEPS', S['h2']))
for txt in [
    'Demo: live training on the same laptop + text generation from the 250K-vocabulary model (~100 words/sec).',
    'Technical report: the 12-section document with all measurements, charts and validation steps (FINAL_REPORT.pdf).',
    'Meeting: 30 minutes for investment proposal, roadmap and go-to-market plan.',
]:
    el.append(Paragraph('• ' + txt, S['body']))

el.append(Spacer(1, 0.5 * cm))
el.append(Paragraph('V3 Project — LLM Training on a 6GB GPU  |  Draft, pre-meeting  |  August 2026',
                    ParagraphStyle('f', parent=S['cap'], fontSize=9, textColor=colors.grey)))

doc.build(el)
size = os.path.getsize(os.path.join(BASE, 'EXECUTIVE_SUMMARY.pdf'))
print(f'PDF written: EXECUTIVE_SUMMARY.pdf ({size/1e6:.1f} MB)')
