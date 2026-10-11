"""Лабораторная 2. Python 3.10+, matplotlib, pillow.

python solution.py                    # два GIF, PNG и JSON с результатами
python solution.py --show             # показать анимации в окнах
python solution.py --input input.json # свои polygon и point

Геометрия работает с точными Fraction, без допуска epsilon. Входной
многоугольник должен быть простым, без повторения первой вершины в конце.
"""
from __future__ import annotations
import argparse
from dataclasses import dataclass
from fractions import Fraction as F
import json
from pathlib import Path
import random

Point = tuple[F, F]
LABELS = {'inside': 'Внутри', 'outside': 'Снаружи',
          'edge': 'На ребре', 'vertex': 'В вершине'}


def point(value) -> Point:
    if len(value) != 2:
        raise ValueError('У точки должны быть две координаты')
    return tuple(F(str(x)) for x in value)


def polygon_points(values) -> list[Point]:
    p = [point(v) for v in values]
    if len(p) < 3 or len(set(p)) != len(p):
        raise ValueError('Нужно минимум три различные вершины; не замыкайте список')
    if sum(a[0]*b[1]-a[1]*b[0] for a, b in zip(p, p[1:]+p[:1])) == 0:
        raise ValueError('Многоугольник должен иметь ненулевую площадь')
    return p


def bounds(poly):
    return (min(x for x, y in poly), max(x for x, y in poly),
            min(y for x, y in poly), max(y for x, y in poly))


def bbox_test(q, poly):
    """Граница габаритов включается: здесь нельзя отвергать точки на ней."""
    lo, hi, bottom, top = bounds(poly)
    x, y = q
    return lo <= x <= hi and bottom <= y <= top


def on_segment(q, a, b):
    x, y = q
    return ((b[0]-a[0])*(y-a[1]) == (b[1]-a[1])*(x-a[0])
            and min(a[0], b[0]) <= x <= max(a[0], b[0])
            and min(a[1], b[1]) <= y <= max(a[1], b[1]))


@dataclass(frozen=True)
class EdgeStep:
    index: int
    counted: bool
    intersection: Point | None
    explanation: str


@dataclass
class Result:
    location: str
    bbox_pass: bool
    steps: list[EdgeStep]
    boundary_index: int | None = None

    @property
    def crossings(self):
        return sum(s.counted for s in self.steps)


def classify(q, poly) -> Result:
    """O(n) по времени. Список шагов O(n), нужен для анимации.

    Луч всегда направлен вправо. Ребро участвует, только если
    min(ya, yb) <= y < max(ya, yb); горизонтальные ребра пропускаются.
    В обычной вершине учитывается одно ребро; в локальном минимуме —
    два, в локальном максимуме — ноль. Последние два случая сохраняют
    чётность. Совпавшие пересечения НЕ объединяются.
    """
    q = point(q)
    poly = polygon_points(poly)
    if not bbox_test(q, poly):
        return Result('outside', False, [])
    # Вершины проверяются раньше рёбер, чтобы различить оба вида границы.
    for i, a in enumerate(poly):
        if q == a:
            return Result('vertex', True, [], i)
    for i, a in enumerate(poly):
        if on_segment(q, a, poly[(i+1) % len(poly)]):
            return Result('edge', True, [], i)
    x, y = q
    steps = []
    for i, a in enumerate(poly):
        b = poly[(i+1) % len(poly)]
        if a[1] == b[1]:
            steps.append(EdgeStep(i, False, None, 'Горизонтальное ребро: не считаем'))
        elif not min(a[1], b[1]) <= y < max(a[1], b[1]):
            steps.append(EdgeStep(i, False, None, 'Уровень вне полуинтервала [ymin, ymax)'))
        else:
            hit_x = a[0] + (y-a[1])*(b[0]-a[0])/(b[1]-a[1])
            counted = hit_x > x
            steps.append(EdgeStep(i, counted, (hit_x, y),
                                  'Пересечение справа: +1' if counted else 'Пересечение слева: +0'))
    return Result('inside' if sum(s.counted for s in steps) % 2 else 'outside', True, steps)


def generate_demo(seed=7):
    """Воспроизводимая генерация: вогнутый шаблон + случайный масштаб/сдвиг.

    Шаблон нужен, чтобы гарантировать горизонтальные рёбра, обычную
    вершину, локальные минимум и максимум на уровне луча.
    """
    rng = random.Random(seed)
    sx, sy = rng.randint(1, 3), rng.randint(1, 3)
    dx, dy = rng.randint(-4, 4), rng.randint(-4, 4)
    def transform(v):
        x, y = point(v)
        return (sx*x+dx, sy*y+dy)
    template = [(0, 0), (8, 0), (9, 2), (8, 6), (6, 6),
                (5, 3), (4, 5), (2, 3), (0, 6)]
    poly = [transform(v) for v in template]
    bbox_cases = [
        ('Левее габаритов', (-1, 2), 'outside'),
        ('Правее габаритов', (10, 2), 'outside'),
        ('Ниже габаритов', (4, -1), 'outside'),
        ('Выше габаритов', (4, 7), 'outside'),
        ('Левая граница габаритов: ребро', (0, 2), 'edge'),
        ('Правая граница габаритов: вершина', (9, 2), 'vertex'),
        ('Нижняя граница габаритов: ребро', (4, 0), 'edge'),
        ('Верхняя граница габаритов: ребро', (7, 6), 'edge'),
        ('Угол габаритов: вершина', (0, 0), 'vertex'),
        ('Угол габаритов вне многоугольника', (9, 6), 'outside'),
        ('В габаритах, но снаружи', (4, '5.5'), 'outside'),
    ]
    ray_cases = [
        ('Обычная точка внутри', (4, 1), 'inside'),
        ('Точка в выемке снаружи', (4, '5.5'), 'outside'),
        ('Луч через обычную вершину: одно ребро', (1, 2), 'inside'),
        ('Луч через локальные минимумы: по два ребра', (1, 3), 'inside'),
        ('Луч через локальный максимум: ноль рёбер', (1, 5), 'outside'),
        ('Луч вдоль горизонтального ребра', (1, 6), 'outside'),
        ('Точка на горизонтальном ребре', (4, 0), 'edge'),
        ('Точка на наклонном ребре', ('5.5', '4.5'), 'edge'),
        ('Точка в вершине выемки', (5, 3), 'vertex'),
        ('Чуть ниже локального минимума', (4, '2.99'), 'inside'),
        ('Чуть выше локального минимума', (5, '3.01'), 'outside'),
    ]
    def cases(rows):
        return [(title, transform(q), expected) for title, q, expected in rows]
    return poly, cases(bbox_cases), cases(ray_cases)


def draw_base(ax, poly, q, result):
    import matplotlib.patches as patches
    xs, ys = zip(*[(float(x), float(y)) for x, y in poly])
    lo, hi, bottom, top = map(float, bounds(poly))
    margin = max(hi-lo, top-bottom)*0.17
    ax.clear()
    ax.add_patch(patches.Polygon(list(zip(xs, ys)), facecolor='#dbeafe',
                                edgecolor='#334155', linewidth=2))
    ax.add_patch(patches.Rectangle((lo, bottom), hi-lo, top-bottom,
                                  fill=False, linestyle='--', color='#8b5cf6', linewidth=1.7))
    ax.scatter(xs, ys, color='#334155', s=22, zorder=4)
    for i, (x, y) in enumerate(zip(xs, ys)):
        ax.annotate(f'p{i+1}', (x, y), xytext=(5, 6), textcoords='offset points', fontsize=9)
    color = {'inside': '#16a34a', 'outside': '#dc2626', 'edge': '#ea580c', 'vertex': '#ea580c'}[result.location]
    ax.scatter([float(q[0])], [float(q[1])], color=color, s=70, zorder=8)
    ax.annotate('p₀', tuple(map(float, q)), xytext=(7, -15), textcoords='offset points', color=color)
    ax.set(xlim=(lo-margin, hi+margin), ylim=(bottom-margin, top+margin), xlabel='x', ylabel='y')
    ax.set_aspect('equal')
    ax.grid(alpha=0.2)


def make_animation(poly, cases, mode, output=None):
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    fig, ax = plt.subplots(figsize=(10, 7))
    fig.subplots_adjust(top=0.81, bottom=0.23)
    heading = fig.text(0.07, 0.94, '', fontsize=15, weight='bold')
    detail = fig.text(0.07, 0.865, '', fontsize=11)
    footer = fig.text(0.07, 0.07, '', fontsize=11, linespacing=1.6)
    results = [classify(q, poly) for _, q, _ in cases]
    frames = []
    for ci, result in enumerate(results):
        # Повтор заключительного кадра даёт время прочитать результат.
        stages = [-1] + (list(range(len(result.steps))) if mode == 'ray' else []) + [len(result.steps)]*3
        frames.extend((ci, stage) for stage in stages)
    def update(frame):
        ci, stage = frames[frame]
        title, q, expected = cases[ci]
        result = results[ci]
        draw_base(ax, poly, q, result)
        heading.set_text(('Габаритный тест' if mode == 'bbox' else 'Лучевой тест') + f' • случай {ci+1}/{len(cases)}')
        detail.set_text(f'{title}\np₀ = ({float(q[0]):g}, {float(q[1]):g})')
        if stage == -1:
            text = ('Этап 1: габаритный тест — ' + ('пройден; нужна проверка границы и луча' if result.bbox_pass else 'не пройден: точка снаружи'))
        elif mode == 'ray' and stage < len(result.steps):
            step = result.steps[stage]
            a, b = poly[step.index], poly[(step.index+1) % len(poly)]
            ax.plot([float(a[0]), float(b[0])], [float(a[1]), float(b[1])], color='#f59e0b', linewidth=4, zorder=5)
            text = f'Ребро p{step.index+1}–p{(step.index+1)%len(poly)+1}: {step.explanation}\nУчтено пересечений: {sum(s.counted for s in result.steps[:stage+1])}'
        else:
            text = f'Результат: {LABELS[result.location]}'
            if not result.bbox_pass:
                text += '. Лучевой тест не запускается.'
            elif result.boundary_index is not None:
                i = result.boundary_index
                if result.location == 'vertex':
                    text += f' p{i+1}. Лучевой тест не требуется.'
                else:
                    text += f' p{i+1}–p{(i+1)%len(poly)+1}. Лучевой тест не требуется.'
            else:
                text += f'. Пересечений: {result.crossings} — ' + ('нечётное число.' if result.crossings%2 else 'чётное число.')
        if mode == 'ray' and result.steps and stage >= 0:
            right = ax.get_xlim()[1]
            ax.annotate('', xy=(right, float(q[1])), xytext=tuple(map(float, q)),
                        arrowprops={'arrowstyle': '->', 'color': '#2563eb', 'lw': 1.8})
            for s in result.steps[:min(stage+1, len(result.steps))]:
                if s.counted:
                    ax.scatter(*map(float, s.intersection), color='#2563eb', s=45, zorder=7)
            # Число учитывает кратность даже при совпадении маркеров в вершине.
        footer.set_text(text + '\nПунктир: габариты. Синий: луч/пересечения. Жёлтый: проверяемое ребро.')
        return []
    anim = FuncAnimation(fig, update, frames=len(frames), interval=550, blit=False, repeat=True)
    if output:
        anim.save(str(output), writer=PillowWriter(fps=2), dpi=90)
    return fig, anim


def snapshot(poly, q, path):
    import matplotlib.pyplot as plt
    result = classify(q, poly)
    fig, ax = plt.subplots(figsize=(8, 6))
    draw_base(ax, poly, q, result)
    ax.set_title(f'{LABELS[result.location]}: p₀ = ({float(q[0]):g}, {float(q[1]):g})\nПересечений: {result.crossings}' if result.steps else LABELS[result.location])
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--out', type=Path, default=Path('results'))
    parser.add_argument('--show', action='store_true')
    parser.add_argument('--input', type=Path, help='JSON: {"polygon": [[x,y],...], "point": [x,y]}')
    args = parser.parse_args()
    if not args.show:
        import matplotlib
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    args.out.mkdir(parents=True, exist_ok=True)
    if args.input:
        data = json.loads(args.input.read_text(encoding='utf-8'))
        poly, q = polygon_points(data['polygon']), point(data['point'])
        result = snapshot(poly, q, args.out/'point.png')
        print(f'{LABELS[result.location]}; габаритный тест: {result.bbox_pass}; пересечений: {result.crossings}')
        fig, anim = make_animation(poly, [('Пользовательская точка', q, None)], 'ray', args.out/'ray_test.gif')
        if args.show:
            plt.show()
        plt.close(fig)
        return
    poly, bbox_cases, ray_cases = generate_demo(args.seed)
    records = []
    animations = []
    for mode, cases in [('bbox', bbox_cases), ('ray', ray_cases)]:
        for title, q, expected in cases:
            r = classify(q, poly)
            if r.location != expected:
                raise AssertionError(f'{title}: {r.location} != {expected}')
            records.append(dict(test=mode, case=title, point=[float(v) for v in q],
                                result=LABELS[r.location], bbox_pass=r.bbox_pass,
                                crossings=r.crossings, expected=LABELS[expected]))
            print(f'{title}: {LABELS[r.location]} (пересечений: {r.crossings})')
        fig, anim = make_animation(poly, cases, mode, args.out/f'{mode}_test.gif')
        animations.append(anim)
        if not args.show:
            plt.close(fig)
    snapshot(poly, ray_cases[3][1], args.out/'example.png')
    (args.out/'cases.json').write_text(json.dumps(dict(seed=args.seed, polygon=[[float(v) for v in p] for p in poly], cases=records), ensure_ascii=False, indent=2), encoding='utf-8')
    if args.show:
        plt.show()


if __name__ == '__main__':
    main()