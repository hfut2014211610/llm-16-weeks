import markdown, weasyprint, sys, os
here = os.path.dirname(os.path.abspath(__file__))
md = open(os.path.join(here, 'handout.md'), encoding='utf-8').read()
# 字体：优先 ~/.fonts，其次项目 work/fonts（静态 Regular/Bold，避免可变字体粗细错乱）
FONT_DIR = next((d for d in [os.path.expanduser('~/.fonts'), os.path.join(here, '..', '..', 'work', 'fonts')]
                 if os.path.exists(os.path.join(d, 'NotoSansSC-Regular.ttf'))), None)
for k, v in [a.split('=',1) for a in sys.argv[1:]]:
    md = md.replace('{{'+k+'}}', v.replace('\\n','\n'))
html = markdown.markdown(md, extensions=['fenced_code','tables','md_in_html','codehilite'],
                         extension_configs={'codehilite':{'noclasses':True,'pygments_style':'friendly'}})
css = (f"""@font-face {{ font-family: 'Noto Sans SC'; src: url('file://{FONT_DIR}/NotoSansSC-Regular.ttf'); font-weight: normal; }}
@font-face {{ font-family: 'Noto Sans SC'; src: url('file://{FONT_DIR}/NotoSansSC-Bold.ttf'); font-weight: bold; }}
""" if FONT_DIR else "") + """
@page { size: A4; margin: 18mm 16mm; @bottom-center { content: counter(page); font-size: 9pt; color:#888; } }
body { font-family: 'Noto Sans SC', sans-serif; font-size: 10.5pt; line-height: 1.65; color:#222; }
h1 { font-size: 20pt; margin: 0 0 2pt; color:#1a3c6e; }
.sub { color:#666; margin-top:0; }
h2 { font-size: 14pt; color:#1a3c6e; border-bottom: 1.5px solid #c9d6ea; padding-bottom:3pt; margin-top: 18pt; page-break-after: avoid; }
hr { border:none; margin: 6pt 0; }
pre, code { font-family: 'DejaVu Sans Mono', 'Noto Sans SC', monospace; font-size: 8.8pt; }
pre { background:#f6f8fa; padding:7pt 9pt; border-radius:4px; white-space: pre-wrap; word-break: break-all; page-break-inside: avoid; }
.codehilite { background:#f6f8fa !important; border-radius:4px; }
p code, li code, td code { background:#eef1f5; padding:0 3px; border-radius:3px; }
table { border-collapse: collapse; width:100%; margin: 6pt 0; font-size: 9.6pt; page-break-inside: avoid; }
th, td { border:1px solid #d0d7de; padding:4pt 6pt; text-align:left; vertical-align: top; }
th { background:#eef3fa; }
.back { background:#fff8e6; border-left: 4px solid #e0a800; padding: 4pt 11pt; margin: 9pt 0; border-radius: 3px; }
.back pre, .back .codehilite { background:#fffdf5 !important; }
img { max-width: 100%; display:block; margin: 6pt auto; page-break-inside: avoid; }
.back img { max-width: 100%; }
.foot { color:#777; font-size:9pt; margin-top:14pt; }
"""
doc = f"<html><head><meta charset='utf-8'><style>{css}</style></head><body>{html}</body></html>"
weasyprint.HTML(string=doc, base_url=here).write_pdf(os.path.join(here, '第2周讲义-MLP激活与BatchNorm.pdf'))
print('ok')
