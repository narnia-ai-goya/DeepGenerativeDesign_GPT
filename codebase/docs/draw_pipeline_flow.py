"""Regenerates docs/pipeline_flow.png — the stage-by-stage data-flow diagram.

Layout: one y unit = one text line. Each stage card's height is computed from its
content, so nothing overflows however much text a stage carries. Edit the STAGES
list below and re-run; the figure resizes itself.

    python docs/draw_pipeline_flow.py

Needs matplotlib only. The labels are Korean, so it needs a Korean-capable font:
set KR_FONT=/path/to/font.ttf, or install one of the usual packages
(fonts-nanum, fonts-noto-cjk). Without one the text renders as boxes and the
script says so.
"""
import os
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib import font_manager as fm
from pathlib import Path

SP = Path(__file__).resolve().parent          # output lands next to this script

_CANDIDATES = [os.environ.get('KR_FONT'),
               SP / 'NotoSansKR.ttf',
               '/usr/share/fonts/truetype/nanum/NanumGothic.ttf',
               '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
               '/usr/share/fonts/truetype/noto/NotoSansKR-Regular.ttf',
               '/usr/share/fonts/truetype/noto/NotoSansCJKkr-Regular.otf',
               Path.home() / '.fonts' / 'NotoSansKR.ttf']
for _c in _CANDIDATES:
    if _c and Path(_c).exists():
        fm.fontManager.addfont(str(_c))
        plt.rcParams['font.family'] = fm.FontProperties(fname=str(_c)).get_name()
        print(f'font: {_c}')
        break
else:
    print('WARNING: no Korean font found — labels will render as boxes.\n'
          '         set KR_FONT=/path/to/font.ttf, or apt install fonts-nanum',
          file=sys.stderr)
MONO = {'family': 'DejaVu Sans Mono'}


def is_mono_safe(s):
    """한글이 없으면 mono 사용 — DejaVu Sans Mono에 한글 글리프가 없어 두부가 된다.
    ·, →, ³ 같은 기호는 mono에도 있으므로 한글 블록만 검사한다."""
    return not any(0xAC00 <= ord(ch) <= 0xD7AF or 0x3130 <= ord(ch) <= 0x318F
                   or 0x1100 <= ord(ch) <= 0x11FF for ch in s)

INK, INK2, INK3 = '#101720', '#3d4b5c', '#7d8c9c'
RULE, RULE2, SUNK, CARD = '#cfd7e0', '#e6ebf0', '#eef2f6', '#ffffff'
BLUE, ORANGE, GREEN, VIOLET = '#2a78d6', '#d95926', '#158c62', '#6b4fc4'
BLUE_BG, ORANGE_BG, GREEN_BG, VIOLET_BG = '#e6effb', '#fdece4', '#e2f4ec', '#ece7fa'

# ── 행 헬퍼: (텍스트, 종류) ─────────────────────────────────────────
def F(t, c=INK2):      return ('f', t, c)      # 파일명 (mono)
def N(t):              return ('n', t, INK3)   # 설명
def H(t):              return ('h', t, INK)    # 처리 단계 제목
def B(t):              return ('b', t, INK2)   # 처리 단계 본문
def S():               return ('s', '', None)  # 여백
def K(t, c=BLUE):      return ('k', t, c)      # 강조 상자


STAGES = [
  dict(badge='STAGE 0', title='입력 준비 — 원본 STL 3개에서 파생물 전부',
       envs=[('CPU', SUNK, INK3), ('fenics', ORANGE_BG, ORANGE)], script='run_prep.py',
       ins=[F('data_real/<dom>/original_DesignSpace.stl'),
            N('설계 영역 — 생성된 부품이 벗어날 수 없는 외곽'), S(),
            F('data_real/<dom>/fixed.stl'), N('고정면 (FEA에서 클램프)'), S(),
            F('data_real/<dom>/load.stl'), N('하중면 (1000 N 작용)'), S(),
            F('configs/<dom>.json → stages.prep'),
            N('remesh edge · 격자 해상도 · bbox margin · dilate')],
       proc=[H('1  remesh   remesh_domain_bc.py'),
             B('원본 CAD STL은 용접되지 않은 삼각형 수프(V=3F).'),
             B('정점 병합 후 등방 remesh → 정삼각형에 가깝게'), S(),
             H('2  voxel   make_bc_proper_pysdf.py'),
             B('pysdf로 복셀 중심을 rasterize → 64³ 격자.'),
             B('고정/하중 영역을 dilate해 BC 채널로 기록'), S(),
             H('3  fea_domain   파일 스테이징'),
             B('in-loop FEM 솔버는 디렉터리 안의 고정된 세 파일명을 찾음 →'),
             B('역할별로 판본을 골라 그 이름으로 복사'), S(),
             H('4  fea_mesh   fenics_fea_bracket.py --build-mesh-only'),
             B('설계 영역은 모든 프롬프트에서 동일 → tet를 한 번만 만들어'),
             B('전 실험이 공유 (실행마다 45회 → 도메인당 3회)')],
       outs=[F('original_DesignSpace_remesh.stl', BLUE),
             F('fixed_remesh.stl · load_remesh.stl', BLUE),
             N('→ Stage C 후처리, Stage D의 BC면, Stage B의 peg면'), S(),
             F('voxel.npz', BLUE),
             N('→ Stage B가 bc_proper + bracket_occ 두 역할로 읽음'), S(),
             F('fea_domain/{original_DesignSpace, fixed, load}.stl', BLUE),
             N('→ Stage B의 in-loop FEM 도메인 (설계영역은 원본, peg는 remesh)'), S(),
             F('fea_shared.msh', BLUE),
             N('→ Stage B의 전 실험이 공유하는 tet')]),

  dict(badge='STAGE A', title='Conditioning — 스타일을 6개 뷰 이미지로',
       envs=[('sd', BLUE_BG, BLUE), ('pyvista', SUNK, INK3)], script='run_conditioning.py',
       ins=[F('data_real/<dom>/*.stl   (원본)'),
            N('렌더·마스크는 기하만 쓰므로 원본 STL 기준'), S(),
            F('configs/curated_prompts.json'),
            N('스타일당 prompt · negative · descriptor · form'), S(),
            F('stages.cond_ref / cond_stylize / cond_img2img'),
            N('모든 수치가 config에서 — 스크립트에 하드코딩 없음')],
       proc=[H('1·2  render + masks   cond_render_pv.py / cond_render_masks.py'),
             B('도메인 기하를 6방향 1024px 렌더 → renders/,  고정·하중 영역의'),
             B('가림 마스크 → masks/.  기하만 쓰므로 도메인당 캐시'), S(),
             H('3  ref   text2img_style_ref.py'),
             B('SDXL t2i(descriptor + ref_common) → 중앙 crop 0.35 → 1024 확대 →'),
             B('256개 문턱값 전수 탐색으로 검정 35% 이진화.'),
             B('결과는 "검정 구조 / 흰 구멍" 두 색 타일 — IP-Adapter가 읽을 형태'), S(),
             H('4  inpaint   cond_stylize_inpaint_refip_xattn.py'),
             B('SDXL inpaint + IP-Adapter(ref) + cross-view attention.'),
             B('마스크 밖 영역에만 스타일을 새김 → remask/'), S(),
             H('5·6  normalize → img2img'),
             B('휘도 정규화(BC 영역은 검정=재료로 채워 보호) → white/ →'),
             B('SDXL img2img(ip_scale 0, strength 0.7 → 실제 28 step)로 6뷰 톤 통일')],
       outs=[F('data/<dom>/conditioning/<style>/v*.png', BLUE),
             N('최종 6뷰: v00_front_lo · v02_right_lo · v04_back_lo ·'),
             N('v06_left_lo · v_top · v_bottom'),
             N('→ Stage B가 읽는 유일한 조건 입력'), S(),
             F('experiments/<ER>/<dom>/cond/…'),
             N('중간 산출물(renders · masks · ref · remask · white)은'),
             N('실험 폴더에 남아 추적 가능'), S(),
             K('image-first 진입점 — Stage A를 건너뛰고 위 경로에'),
             K('6뷰를 직접 두면 됨. 이후 단계는 완전히 동일.'),
             K('두 진입점이 여기서 만난다.')]),

  dict(badge='STAGE B', title='생성 — 확산 과정에 물리를 개입',
       envs=[('direct3ds2 · GPU', VIOLET_BG, VIOLET), ('fenics 서브프로세스', ORANGE_BG, ORANGE)],
       script='generate_with_physics_guidance.py',
       ins=[F('conditioning/<style>/v*.png'),
            N('Stage A 산출 · 형상의 "모양"을 결정'), S(),
            F('voxel.npz'), N('Stage 0 산출 · 설계 영역 + BC 격자'), S(),
            F('fea_domain/ + fea_shared.msh'),
            N('Stage 0 산출 · in-loop FEM의 도메인과 tet'), S(),
            F('stages.mesh   (~40개 값)'),
            N('seed 42 · --deterministic · FEA_NORMALIZE=0')],
       proc=[H('1  dense 64³ cascade'),
             B('latent에 직접 gradient guidance — 영역 밖 억제 out_w 30, BC 유지'),
             B('bc_w 10, 부피·대칭·두께·최소반경, 그리고 FEM compliance'),
             B('fea_w (warmup 0.3 이후 개입, SIMP penal 2) — 도메인별:'),
             B('bracket 1.5e-12 · motor_mount 7.5e-12 · link 4.5e-14'), S(),
             H('2  in-loop FEM   fenics_fea_bracket.py  (서브프로세스)'),
             B('fea_shared.msh에 현재 밀도를 실어 선형탄성을 풀고'),
             B('compliance의 민감도를 돌려줌. 하중 방향은 LOAD_MODE로 전달'),
             B('— config가 결정하며, stages.fea.force_dir와 다르면 실행 전 중단'), S(),
             H('3  sparse 512³ cascade   (50 step)'),
             B('표면 법선/Laplacian 정칙화 sp_lap_w 50, 내부 채움 sp_interior_w 10,'),
             B('최소 반경 sp_rmin_w, FEM 항 sp_fea_w — 도메인별:'),
             B('bracket 3e-11 · motor_mount 1.5e-10 · link 9e-13')],
       outs=[F('mesh.obj', BLUE),
             N('생성된 원 형상 (peg도 설계영역 clip도 아직 없음) → Stage C'), S(),
             F('mesh.glb · mesh_dense.obj', BLUE),
             N('뷰어용 / dense 단계 중간 결과'), S(),
             F('gen.log · loss csv', BLUE),
             N('step별 fea_comp · l_rmin · l_thick · l_vol … 추적'), S(),
             K('Ablation — run_ablation.py가 같은 스테이지 함수를', INK2),
             K('재사용해 abl/<variant>/에 기록.', INK2),
             K('off = FEM 항만 제거, 나머지는 전부 동일.', INK2)]),

  dict(badge='STAGE C', title='후처리 — 조립 가능한 부품으로',
       envs=[('direct3ds2', VIOLET_BG, VIOLET)], script='stages.post',
       ins=[F('mesh.obj'), S(),
            F('*_remesh.stl'), N('고정 · 하중 · 설계영역 —'),
            N('Stage 0 산출을 여기서 처음 씀'), S(),
            F('stages.post'),
            N('peg dilate · edge_mm · laplacian_iters/lamb')],
       proc=[H('1  union + clip   post_hybrid_union_clip.py'),
             B('manifold3d로 생성 형상 ∪ 고정/하중 peg를 메시 수준에서 합집합'),
             B('(peg 목둘레는 국소 marching cubes로 확장), 이어서 설계 영역으로 clip.'),
             B('본체는 재샘플링하지 않음 — 생성된 표면이 그대로 남는다'), S(),
             H('2  remesh + smooth   surface_remesh_pre.py'),
             B('등방 remesh(edge_mm) + Laplacian smoothing —'),
             B('boolean이 남긴 sliver 삼각형과 주름 제거')],
       outs=[F('hybrid.obj', BLUE),
             N('boolean 직후 중간 결과 (실측 face: bracket 47만 ·'),
             N('motor_mount 84만 · link 4.6만).'),
             N('다음 단계의 입력일 뿐 — 검증에는 쓰지 않음'), S(),
             F('final.obj    ← 배포 형상', GREEN),
             N('파이프라인이 실제로 산출하는 부품'),
             N('→ Stage D의 검증 대상, 결과 그림에 실리는 메시')]),

  dict(badge='STAGE D', title='검증 — 사후 FEA',
       envs=[('fenics', ORANGE_BG, ORANGE)], script='fea_prep_and_run.py',
       ins=[F('final.obj', GREEN), N('배포 형상을 그대로 검증   ← 이번 변경'), S(),
            F('fixed_remesh.stl · load_remesh.stl'),
            N('원본 CAD peg는 삼각형이 너무 커서 솔버의 6 mm'),
            N('근접 판정을 통과 못 함 (link는 peg 면의 약 29%를'),
            N('놓침) → remesh판을 씀'), S(),
            F('stages.fea'),
            N('E 110 GPa · ν 0.3 · 1000 N · force_dir · tet_size')],
       proc=[H('1  표면 정리   pymeshlab + pymeshfix'),
             B('등방 remesh(target_edge_mm — bracket·mm 2.5 mm, link 1.5 mm) →'),
             B('정점 병합 · 비다양체 복구 · 구멍 닫기 →'),
             B('최대 성분만 남겨 watertight 보장.'),
             B('quadric decimation은 쓰지 않음 — 얇은 격자 부재가 사라지므로'), S(),
             H('2  tet + 해석   fea_tet_from_mesh.py  (서브프로세스)'),
             B('gmsh HXT로 4면체 생성 → DOLFINx 선형탄성.'),
             B('고정면 클램프, 하중면에 1000 N 분포.'),
             B('실패 시 escalation: HXT → Delaunay → edge 확대 → repair fallback'), S(),
             H('※ pymeshlab과 gmsh/dolfinx는 한 프로세스에 공존 불가'),
             B('libstdc++ 심볼 충돌 — 프로세스 분리가 이 단계의 전제.'),
             B('FENICS_PY가 잘못되면 시작 시 import 프로브가 즉시 중단')],
       outs=[F('fea_input_clean.stl', BLUE),
             N('해석에 실제로 들어간 표면 — 재현·디버깅용'), S(),
             F('fea_tet_summary.json', GREEN),
             N('compliance · vm_max · u_max · n_tet_cells · force_dir'),
             N('compliance > 1 J 이면 퇴화한 tet으로 보고 실패 처리'), S(),
             K('refea.py — 값이 없거나 범위를 벗어난 실행만 골라', INK2),
             K('직렬로 재계산. gmsh 병렬 실행이 불안정하기 때문.', INK2)]),
]

# ── 크기 계산 ──────────────────────────────────────────────────────
LH = 1.0                     # 한 줄 = 1 단위
HEADH, PADT, PADB = 2.4, 1.9, 1.2
GAPS = 2.6                   # 단계 사이 간격
SGAP = 0.55                  # S() 여백 크기


def col_units(rows):
    return sum(SGAP if r[0] == 's' else LH for r in rows)


heights = []
for st in STAGES:
    body = max(col_units(st['ins']), col_units(st['proc']), col_units(st['outs']))
    heights.append(HEADH + PADT + body + PADB)

TOP_UNITS = 7.2                                     # 머리말
TOTAL = TOP_UNITS + sum(heights) + GAPS * (len(STAGES) - 1) + 1.6
UNIT_IN = 0.168
FIG_W = 20.6
fig = plt.figure(figsize=(FIG_W, TOTAL * UNIT_IN), dpi=130, facecolor='white')
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 100); ax.set_ylim(0, TOTAL); ax.axis('off')

X0, XW = 2.6, 94.4
CW = (24.0, 42.0, 28.4)                             # 열 폭
C1 = X0 + CW[0]
C2 = C1 + CW[1]


def txt(x, y, s, size, color=INK, weight='normal', ha='left', mono=False, z=5):
    kw = dict(MONO) if mono else {}
    return ax.text(x, y, s, fontsize=size, color=color, fontweight=weight, ha=ha,
                   va='center', zorder=z, **kw)


def draw_rows(x, y, rows, kw_width=None):
    """행 리스트를 위에서 아래로 렌더."""
    for kind, s, c in rows:
        if kind == 's':
            y -= SGAP; continue
        if kind == 'f':
            txt(x, y, s, 8.4, c, 'bold', mono=is_mono_safe(s))
        elif kind == 'n':
            txt(x, y, s, 7.7, c)
        elif kind == 'h':
            txt(x, y, s, 8.7, INK, 'bold')
        elif kind == 'b':
            txt(x + 1.4, y, s, 8.0, c)
        elif kind == 'k':
            t = txt(x, y, s, 7.7, c)
            t.set_bbox(dict(boxstyle='round,pad=0.22', fc=BLUE_BG if c == BLUE else SUNK,
                            ec='none'))
        y -= LH
    return y


def chip(x, y, s, fc, tc):
    t = txt(x, y, s, 7.4, tc, 'bold', ha='right', mono=is_mono_safe(s))
    t.set_bbox(dict(boxstyle='round,pad=0.30', fc=fc, ec='none'))
    return t


# ══════════ 머리말 ══════════
yc = TOTAL - 1.4
txt(X0, yc, 'SUPPLEMENTARY · 파이프라인 데이터 흐름', 9.4, INK3, 'bold')
txt(X0, yc - 2.1, '무엇이 나오고, 어디로 가서, 어떻게 되는가', 21, INK, 'bold')
txt(X0, yc - 4.3, '원본 CAD STL 3개에서 검증된 compliance까지.    '
                  '파란 글자 = 그 단계가 만든 파일,   초록 = 배포 형상과 최종 결과', 9.2, INK2)
ax.plot([X0, X0 + XW], [yc - 5.4, yc - 5.4], color=INK, lw=1.7, zorder=1)

# ══════════ 단계 카드 ══════════
cy = TOTAL - TOP_UNITS
bounds = []
for st, h in zip(STAGES, heights):
    top, bot = cy, cy - h
    bounds.append((top, bot))
    ax.add_patch(FancyBboxPatch((X0, bot), XW, h, boxstyle='round,pad=0,rounding_size=0.5',
                                fc=CARD, ec=RULE, lw=1.1, zorder=2))
    ax.add_patch(FancyBboxPatch((X0, top - HEADH), XW, HEADH,
                                boxstyle='round,pad=0,rounding_size=0.5',
                                fc=SUNK, ec=RULE, lw=1.0, zorder=3))
    hy = top - HEADH / 2
    t = txt(X0 + 1.0, hy, st['badge'], 8.4, 'white', 'bold', mono=True, z=6)
    t.set_bbox(dict(boxstyle='round,pad=0.34', fc=INK, ec='none'))
    txt(X0 + 9.2, hy, st['title'], 13.2, INK, 'bold', z=6)
    ex = X0 + XW - 1.0
    txt(ex, hy, st['script'], 8.5, INK3, ha='right', mono=True, z=6)
    ex -= len(st['script']) * 0.52 + 2.0
    for name, fc, tc in reversed(st['envs']):
        chip(ex, hy, name, fc, tc)
        ex -= len(name) * 0.55 + 1.8
    for cx in (C1, C2):
        ax.plot([cx, cx], [bot + 0.6, top - HEADH], color=RULE2, lw=1.0, zorder=3)
    ly = top - HEADH - 1.0
    for lx, lab in ((X0 + 1.1, '입력'), (C1 + 1.2, '처리'), (C2 + 1.2, '산출')):
        txt(lx, ly, lab, 7.8, INK3, 'bold', mono=True)
    ry = ly - PADT + 0.4
    draw_rows(X0 + 1.1, ry, st['ins'])
    draw_rows(C1 + 1.2, ry, st['proc'])
    draw_rows(C2 + 1.2, ry, st['outs'])
    cy = bot - GAPS

# ══════════ 단계 간 화살표 ══════════
CX = X0 + XW * 0.42
for i in range(len(bounds) - 1):
    ax.add_patch(FancyArrowPatch((CX, bounds[i][1]), (CX, bounds[i + 1][0] + 0.1),
                                 arrowstyle='-|>', mutation_scale=15, color='#b3c0ce',
                                 lw=2.2, zorder=1))

# ══════════ Stage 0 산출물 재사용 레일 ══════════
RAIL = X0 + XW + 1.5
targets = [(bounds[2][0] - (bounds[2][0] - bounds[2][1]) * 0.5, 'voxel.npz · fea_domain/ · fea_shared.msh'),
           (bounds[3][0] - (bounds[3][0] - bounds[3][1]) * 0.5, '*_remesh.stl'),
           (bounds[4][0] - (bounds[4][0] - bounds[4][1]) * 0.5, 'fixed/load_remesh.stl')]
ax.plot([RAIL, RAIL], [targets[-1][0], bounds[0][1] + 1.2], color=BLUE, lw=1.3,
        ls=(0, (4, 3)), zorder=1)
ax.add_patch(FancyArrowPatch((RAIL, bounds[0][1] + 1.2), (X0 + XW + 0.3, bounds[0][1] + 1.2),
                             arrowstyle='<|-', mutation_scale=10, color=BLUE, lw=1.3, zorder=1))
for yy, s in targets:
    ax.add_patch(FancyArrowPatch((RAIL, yy), (X0 + XW + 0.3, yy), arrowstyle='-|>',
                                 mutation_scale=10, color=BLUE, lw=1.3, zorder=1))
    ax.text(RAIL + 0.7, yy, s, fontsize=7.4, color=BLUE, va='center', ha='left',
            rotation=90, rotation_mode='anchor', **MONO)
ax.text(RAIL + 0.7, bounds[0][1] + 1.2, 'Stage 0 산출물은 이후 세 단계가 계속 읽는다',
        fontsize=7.6, color=BLUE, va='center', ha='left', rotation=90, rotation_mode='anchor')

fig.savefig(SP / 'pipeline_flow.png', facecolor='white', bbox_inches='tight', pad_inches=0.20)
print('saved', SP / 'pipeline_flow.png', f'({FIG_W:.1f} x {TOTAL*UNIT_IN:.1f} in)')
