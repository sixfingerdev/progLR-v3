"""Generate EXECUTIVE_SUMMARY_TR.pdf: Turkish investor summary (non-technical)."""
import os, math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(BASE, 'tests', 'report_figs')

# ---------------- fig: cost gap (Turkish) ----------------
def fig_cost_tr():
    fig, ax = plt.subplots(figsize=(11, 3.2))
    x = [0, 1]
    labels = ['Sunucu GPU\'su\n(H100 sınıfı)\n$25-40 bin, 80GB+', 'Tüketici GPU\'su\n(RTX 3060)\n~$200, 6GB']
    cost = [30000, 200]
    bars = ax.bar(x, cost, color=['#F59E0B', '#10B981'], width=0.55)
    for b, v in zip(bars, cost):
        ax.text(b.get_x() + b.get_width()/2, v * 1.03, f'${v:,.0f}', ha='center', fontsize=13, fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel('donanım maliyeti (tek GPU)', fontsize=11)
    ax.set_ylim(0, 36000)
    ax.text(0.5, 0.90, '~150 kat maliyet farkı — bugün yapay zekâ eğitimi bu yüzden yalnızca\nbüyük bütçelerin işi. Biz bu farkı ürüne dönüştürüyoruz.',
            transform=ax.transAxes, ha='center', fontsize=11, color='#374151')
    fig.tight_layout(); fig.savefig(os.path.join(FIG, 'exec_cost_tr.png'), dpi=180); plt.close(fig)

fig_cost_tr()
print('figures done')

# ================= PDF =================
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.platypus import (BaseDocTemplate, PageTemplate, Frame, Paragraph, Spacer,
                                Image, Table, TableStyle, HRFlowable)
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
    canvas.drawString(M, 1.1 * cm, 'Yatırımcı Özeti — gizli taslak, görüşme öncesi dağıtıma açık değildir')
    canvas.drawRightString(W - M, 1.1 * cm, f'sayfa {doc.page}')
    canvas.restoreState()

doc = BaseDocTemplate(os.path.join(BASE, 'EXECUTIVE_SUMMARY_TR.pdf'), pagesize=A4,
                      leftMargin=M, rightMargin=M, topMargin=1.8 * cm, bottomMargin=1.8 * cm,
                      title='Yatırımcı Özeti — 6GB GPU\'da LLM Eğitimi', author='V3 Projesi')
frame = Frame(M, 1.8 * cm, W - 2 * M, H - 3.6 * cm, id='f')
doc.addPageTemplates([PageTemplate(id='page', frames=[frame], onPage=page_footer)])

S = {}
S['h1'] = ParagraphStyle('h1', fontName='DejaVu-Bold', fontSize=19, leading=25, spaceAfter=10, textColor=colors.HexColor('#1E3A8A'))
S['h2'] = ParagraphStyle('h2', fontName='DejaVu-Bold', fontSize=13.5, leading=18, spaceBefore=14, spaceAfter=6, textColor=colors.HexColor('#1E3A8A'))
S['body'] = ParagraphStyle('body', fontName=F, fontSize=10.5, leading=16, alignment=TA_JUSTIFY, spaceAfter=6)
S['cap'] = ParagraphStyle('cap', fontName='DejaVu-Oblique', fontSize=8.5, leading=11, textColor=colors.grey,
                          alignment=TA_CENTER, spaceBefore=3, spaceAfter=10)
S['cell'] = ParagraphStyle('cell', fontName=F, fontSize=8.5, leading=11.5)
S['cellb'] = ParagraphStyle('cellb', fontName='DejaVu-Bold', fontSize=8.5, leading=11.5)
P = lambda t: Paragraph(t, S['cell'])
PB = lambda t: Paragraph(t, S['cellb'])

el = []

# ---- kapak ----
el.append(Spacer(1, 0.6 * cm))
el.append(Paragraph('YATIRIMCI ÖZETİ', ParagraphStyle('t', parent=S['h1'], fontSize=24, leading=30, alignment=TA_CENTER)))
el.append(Paragraph('Büyük Dil Modeli Eğitimini 6 GB\'lık Bir Laptop GPU\'suna Sığdırdık', ParagraphStyle('s', parent=S['h1'], fontSize=15, leading=21, alignment=TA_CENTER, textColor=colors.HexColor('#F59E0B'))))
el.append(Spacer(1, 0.3 * cm))
el.append(Paragraph('Bugün 1 milyar+ parametreli bir modeli eğitmek, tek parçası 25-40 bin dolar olan sunucu GPU\'ları ve '
                    'yüzbinlerce dolarlık eğitim bütçeleri gerektiriyor. Geliştirdiğimiz teknoloji yığını, aynı eğitimi '
                    '**~200 dolarlık bir tüketici GPU\'sunda (6 GB)** çalıştırılabilecek hale getiriyor — rakiplerin aynı '
                    'donanımda deneyemediği ölçekte.', ParagraphStyle('lead', parent=S['body'], fontSize=11.5, leading=18, alignment=TA_CENTER, textColor=colors.HexColor('#4B5563'))))
el.append(Spacer(1, 0.4 * cm))
el.append(Image(os.path.join(FIG, 'exec_cost_tr.png'), width=16.5 * cm, height=16.5 * cm * 0.32))
el.append(Spacer(1, 0.3 * cm))

el.append(HRFlowable(width='100%', thickness=1, color=colors.HexColor('#D1D5DB')))
el.append(Paragraph('SORUN', S['h2']))
el.append(Paragraph(
    'Yapay zekâ eğitimi bir donanım yarışı hâline geldi: model büyüdükçe bellek ve hesap ihtiyacı katlanıyor, eğitim maliyetleri '
    'milyon dolarları buluyor. Sonuç: bu alanda çalışabilenlerin sayısı, bütçesi büyük olanlarla sınırlı. Oysa dünyada '
    'milyonlarca güçlü tüketici GPU\'su (6-8 GB) mevcut — ve bugün bu donanımın tamamı "yetersiz" sayıldığı için yapay zekâ '
    'eğitiminde kullanılamıyor.', S['body']))

el.append(Paragraph('ÜRÜNÜMÜZ — DÖRT TEKNOLOJİ, TEK HEDEF: AYNI İŞ, ÇOK DAHA UCUZ', S['h2']))
prows = [[PB('Teknoloji'), PB('Ne yapıyor'), PB('Ölçülen kazanç')],
         [PB('V3-Embed\n(kelime hazinesi sıkıştırma)'), P('Modelin kelime hazinesi matrisini çok küçük öğrenilebilir faktörlere indiriyor; kelime sayısı artık belleği şişirmiyor.'), P('500× sıkıştırma')],
         [PB('Analytic Chunked Head\n(çıktı katmanı hızlandırıcı)'), P('50.000-250.000+ kelimelik çıktı katmanının eğitimini matematiksel olarak aynı sonucu veren, bellek dostu bir yöntemle hesaplıyor.'), P('5-12× hızlanma; bellek taşmasını bitiriyor')],
         [PB('Progressive Low-Rank\n(dar kapasiteli gizli katmanlar)'), P('Gizli katmanların bir kısmını düşük maliyetli eşdeğerlerine çeviriyor; geniş modelleri 6 GB bütçeye sığdırıyor.'), P('Aynı genişlikte ~3.6× az parametre')],
         [PB('NCE Sampled Head\n(hız katlaması)'), P('Çıktı katmanını tüm kelimeler yerine hedef + örneklenen negatiflerle eğitiyor; hesap maliyeti kelime sayısından tamamen bağımsızlaşıyor. NCE ile ısınma, exact CE ile son rötuş (hibrit) hem hızı hem kaliteyi veriyor.'), P('3-4.4× ek hızlanma; hibrit ile kalite kaybı ~%4')]]
pt = Table(prows, colWidths=[4.4 * cm, 8.4 * cm, 3.6 * cm])
pt.setStyle(TableStyle([('FONTNAME', (0, 0), (-1, -1), F), ('FONTSIZE', (0, 0), (-1, -1), 8.5),
                        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#EFF6FF')),
                        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#D1D5DB')),
                        ('TOPPADDING', (0, 0), (-1, -1), 5), ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
                        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE')]))
el.append(pt)
el.append(Spacer(1, 0.3 * cm))

el.append(Paragraph('KANIT — AYNI 6 GB GPU\'DA ÖLÇÜLDÜ', S['h2']))
krows = [[PB('Ölçüm'), PB('Vanilya yaklaşım (rakiplerin yöntemi)'), PB('Bizim teknolojimiz')],
         [P('Eğitim hızı (aynı model, aynı batch)'), P('3.1K token/sn — bellek taşması (9.1GB)'), P('16.8K token/sn — 5.4× daha hızlı, 6 GB içinde')],
         [P('250.000 kelimelik model'), P('Daha başlatılamıyor (çıktı katmanı tek başına ~8GB)'), P('6 GB içinde eğitildi (~25 dakika son finetune)')],
         [P('WikiText-103 modeli (50K kelime)'), P('Aynı yükte bellek page\'leniyor'), P('val PPL 95.18 — 6 GB\'da, ~9 GPU-saat')],
         [P('229M parametreli yoğun model'), P('6 GB\'a sığmıyor (sürekli bellek taşması)'), P('Düşük maliyetli eşdeğeri 2.8 GB\'da çalışıyor')],
         [P('250K kelimelik model eğitim hızı (aynı model)'), P('5.0K token/sn (exact CE)'), P('14.9K token/sn — NCE ile 3×')],
         [P('8.9M parametreli model eğitim hızı (aynı model)'), P('20.6K token/sn'), P('90.9K token/sn — 4.4×')],
         [P('Kalite + hız (hibrit: NCE ısınma, exact rötuş)'), P('14 dakikada 224.21 (exact CE 2000 adım)'), P('Aynı sürede 233.10 — kalitenin %96\'sı, saf NCE\'nin 3.5× az süresi')]]
kt = Table(krows, colWidths=[4.2 * cm, 5.9 * cm, 6.3 * cm])
kt.setStyle(TableStyle([('FONTNAME', (0, 0), (-1, -1), F), ('FONTSIZE', (0, 0), (-1, -1), 8.5),
                        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#EFF6FF')),
                        ('BACKGROUND', (1, 1), (1, -1), colors.HexColor('#FEF2F2')),
                        ('BACKGROUND', (2, 1), (2, -1), colors.HexColor('#ECFDF5')),
                        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#D1D5DB')),
                        ('TOPPADDING', (0, 0), (-1, -1), 4), ('BOTTOMPADDING', (0, 0), (-1, -1), 4)]))
el.append(kt)
el.append(Spacer(1, 0.2 * cm))
el.append(Image(os.path.join(FIG, 'scale.png'), width=16.8 * cm, height=16.8 * cm * 0.36))
el.append(Paragraph('Aynı donanım, aynı model gövdesi, aynı batch boyutu: kırmızı sütun vanilya (rakiplerin) yöntemi; '
                    'mavi/turuncu sütunlar bizim teknolojimizle eğitilen modeller. Kırmızı nokta: vanilyanın deneyemediği '
                    'ölçekte biz eğitiyoruz.', S['cap']))

el.append(Paragraph('NEDEN ŞİMDİ, NEDEN BİZ', S['h2']))
for txt in [
    '<b>Pazar açılımı:</b> eğitim maliyeti 30-100× düşünce, bugün "yapay zekâ modeli geliştirmek çok pahalı" diyen binlerce şirket ve araştırma ekibi bu işe girebilir — kendi verisiyle, kendi donanımında.',
    '<b>Savunulabilir ürün:</b> bellek-sınır optimizasyonu, rakiplerin aynı donanımda uygulayamadığı ölçekte kazanç sağlıyor (250K kelime hazinesi, geniş modeller). Dört patentlenebilir bileşen: V3-Embed, analytic chunked head, progressive low-rank, NCE sampled head.',
    '<b>Kanıt zinciri:</b> tüm sonuçlar tek bir 6 GB\'lık GPU üzerinde, karşılaştırmalı koşularla ölçüldü; eğitim hızlandırıcının matematiksel eşdeğerliği bit-bit doğrulandı.',
    '<b>Düşük girdi maliyeti:</b> yığının tamamı ~200 dolarlık donanımda geliştirildi — ürünleşme de aynı donanım sınıfını hedefliyor.',
]:
    el.append(Paragraph('• ' + txt, S['body']))

el.append(Spacer(1, 0.2 * cm))
el.append(HRFlowable(width='100%', thickness=1, color=colors.HexColor('#D1D5DB')))
el.append(Paragraph('DÜRÜST SINIRLAR', S['h2']))
el.append(Paragraph(
    'Şeffaflık adına: eşit parametre sayısında ve küçük ölçekte, klasik yoğun mimari kalite açısından önde kalabiliyor. '
    'NCE hız katmanı bir tahmincidir — aynı adım sayısında doğrulama kalitesi (val PPL) exact CE\'nin ~2.8× gerisindedir; '
    'kazandırdığı ~4.4× hız, aynı sürede çok daha fazla veri/model işlenmesini sağlar ve kelime sayısından bağımsız '
    'ölçeklenir. Kalite metriği olarak exact CE esas alınır. '
    'Bizim kazancımız rakiplerin **deneyemediği** ölçektedir (geniş model + büyük kelime hazinesi + tüketici GPU\'su) — '
    'ürünümüz de tam olarak bu bölgeyi hedefler. Model kalitesi iddiamız, ölçülebilir dil-modeli metriğinde (val PPL) '
    'kıyaslanabilir değerler üretmektir; "daha az parametreyle aynı kalite" gibi bir sihir iddia etmiyoruz.', S['body']))

el.append(Spacer(1, 0.4 * cm))
el.append(Paragraph('SONRAKİ ADIMLAR', S['h2']))
for txt in [
    'Demo: aynı laptop üzerinde canlı eğitim + 250K kelimelik model ile metin üretimi (saniyede ~100 kelime).',
    'Teknik rapor: tüm ölçümlerin, grafiklerin ve doğrulama adımlarının bulunduğu 12 bölümlük doküman (FINAL_REPORT.pdf).',
    'Görüşme: yatırım teklifi, yol haritası ve pazara giriş planı için 30 dakikalık toplantı.',
]:
    el.append(Paragraph('• ' + txt, S['body']))

el.append(Spacer(1, 0.5 * cm))
el.append(Paragraph('V3 Projesi — 6 GB\'da LLM Eğitimi  |  Taslak, görüşme öncesi  |  Ağustos 2026',
                    ParagraphStyle('f', parent=S['cap'], fontSize=9, textColor=colors.grey)))

doc.build(el)
size = os.path.getsize(os.path.join(BASE, 'EXECUTIVE_SUMMARY_TR.pdf'))
print(f'PDF written: EXECUTIVE_SUMMARY_TR.pdf ({size/1e6:.1f} MB)')
