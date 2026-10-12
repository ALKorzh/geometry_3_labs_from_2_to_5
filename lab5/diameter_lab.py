#!/usr/bin/env python3
"""Лабораторная 5: движение точек, оболочка Джарвиса, диаметр множества.

python diameter_lab.py
python diameter_lab.py --mode boundary
python diameter_lab.py --save diameter_demo.gif
python diameter_lab.py --test

На каждом кадре строится оболочка Джарвиса, затем вращающимися калиперами
вычисляется самая удалённая пара. При расстоянии строго больше D только
у этой пары V заменяется на -V. При равных диаметрах выбирается пара
с наименьшими исходными индексами. Пространственных стенок нет.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import numpy as np


def cross(a, b):
    return float(a[0] * b[1] - a[1] * b[0])


def turn(a, b, c):
    return cross(b - a, c - a)


def as_points(points):
    p = np.asarray(points, dtype=float)
    if p.shape == (0,):
        return np.empty((0, 2), dtype=float)
    if p.ndim != 2 or p.shape[1] != 2 or not np.isfinite(p).all():
        raise ValueError('Точки должны образовывать конечный массив (n, 2)')
    return p.copy()


def jarvis_hull(points):
    """Обход Джарвиса за O(nh); вернуть исходные индексы вершин.

    Начало — самая левая, затем нижняя точка. На каждом шаге выбирается
    следующая опорная вершина. Коллинеарные промежуточные точки удаляются.
    Обход против часовой стрелки; начальная вершина в конце не повторяется.
    Повторы координат представляются точкой с минимальным исходным индексом.
    """
    p = as_points(points)
    if len(p) == 0:
        return np.empty(0, dtype=int)
    unique = {}
    for i, s in enumerate(p):
        unique.setdefault(tuple(s), i)
    representatives = list(unique.values())
    if len(representatives) == 1:
        return np.array(representatives, dtype=int)
    start = min(representatives, key=lambda i: (p[i, 0], p[i, 1]))
    current, hull = start, []
    while True:
        hull.append(current)
        candidate = next(i for i in representatives if i != current)
        for i in representatives:
            if i == current or i == candidate:
                continue
            direction = turn(p[current], p[candidate], p[i])
            if (direction < 0 or
                    (direction == 0 and np.dot(p[i] - p[current], p[i] - p[current])
                     > np.dot(p[candidate] - p[current], p[candidate] - p[current]))):
                candidate = i
        current = candidate
        if current == start:
            break
        if len(hull) > len(representatives):
            raise ArithmeticError('Обход Джарвиса не замкнулся')
    return np.array(hull, dtype=int)


def set_diameter(points, hull=None):
    """Диаметр по оболочке: вращающиеся калиперы O(h).

    Вернуть (квадрат диаметра, пару исходных индексов). Для n<2 пара None.
    Противолежащая вершина ищется по максимальной площади треугольника
    с очередной стороной оболочки. Параллельные стороны дают две вершины:
    обе рассматриваются, чтобы не пропустить равные максимумы.
    """
    p = as_points(points)
    if len(p) < 2:
        return 0.0, None
    ids = jarvis_hull(p) if hull is None else np.asarray(hull, dtype=int)
    h = len(ids)
    if h == 1:
        return 0.0, (0, 1)
    best, pair = -1.0, None

    def consider(a, b):
        nonlocal best, pair
        i, k = int(ids[a]), int(ids[b])
        if i == k:
            return
        candidate = tuple(sorted((i, k)))
        d = p[i] - p[k]
        value = float(np.dot(d, d))
        if value > best or (value == best and (pair is None or candidate < pair)):
            best, pair = value, candidate

    if h == 2:
        consider(0, 1)
        return best, pair
    j = 1
    for i in range(h):
        ni = (i + 1) % h
        def area(k):
            return abs(turn(p[ids[i]], p[ids[ni]], p[ids[k]]))
        count = 0
        while area((j + 1) % h) > area(j):
            j = (j + 1) % h
            count += 1
            if count > h:
                raise ArithmeticError('Калиперы: некорректная оболочка')
        consider(i, j)
        consider(ni, j)
        if area((j + 1) % h) == area(j):
            consider(i, (j + 1) % h)
            consider(ni, (j + 1) % h)
    return best, pair


@dataclass(frozen=True)
class Snapshot:
    positions: np.ndarray
    velocities: np.ndarray
    hull: np.ndarray
    pair: tuple[int, int] | None
    diameter: float
    reversed_pair: bool


def simulate(points, velocities, threshold, frames=160, dt=1.0):
    """Измерить диаметр, применить разворот, сохранить состояние кадра.

    Первый кадр проверяет начальные координаты. Далее S += dt*V, затем
    новая оболочка/диаметр и проверка D. Строгое сравнение выполняется по
    квадратам расстояний, без допуска. Условие применяется на каждом кадре,
    даже если диаметр оставался выше D и на предыдущем кадре.
    """
    p, v = as_points(points), as_points(velocities)
    if p.shape != v.shape:
        raise ValueError('На каждую точку нужен один вектор скорости')
    if not np.isfinite(threshold) or threshold < 0:
        raise ValueError('D должен быть конечным и неотрицательным')
    if frames < 1 or not np.isfinite(dt) or dt <= 0:
        raise ValueError('frames >= 1 и dt > 0')
    history = []
    for frame in range(frames):
        if frame:
            p += dt * v
        hull = jarvis_hull(p)
        squared, pair = set_diameter(p, hull)
        reverse = pair is not None and squared > threshold * threshold
        if reverse:
            v[list(pair)] *= -1
        history.append(Snapshot(p.copy(), v.copy(), hull.copy(), pair,
                                float(np.sqrt(squared)), reverse))
    return history


def demo_data(seed=29):
    rng = np.random.default_rng(seed)
    angles = rng.uniform(0, 2 * np.pi, 16)
    radius = rng.uniform(.4, 2., 16)
    points = np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])
    directions = rng.uniform(0, 2 * np.pi, len(points))
    velocities = .075 * np.column_stack([np.cos(directions), np.sin(directions)])
    return points, velocities, 5.0


BOUNDARY_CASES = [
    ('1. Пустое множество', [], [], 2.),
    ('2. Одна точка', [[0., 0.]], [[.08, .04]], 2.),
    ('3. Расстояние точно D', [[0., 0.], [2., 0.]], [[.03125, .015625], [.03125, .015625]], 2.),
    ('4. Расстояние больше D', [[0., 0.], [3., 0.]], [[-.08, 0.], [.08, 0.]], 2.),
    ('5. Совпадающие точки, D = 0', [[0., 0.], [0., 0.], [0., 0.]],
     [[.05, 0.], [.05, 0.], [.05, 0.]], 0.),
    ('6. Коллинеарные точки', [[-2., 0.], [-1., 0.], [0., 0.], [2., 0.]],
     [[-.05, 0.], [0., 0.], [0., 0.], [.05, 0.]], 4.),
]


def make_animation(mode='demo', frames=160, interval=80):
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from matplotlib.patches import Polygon
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10})
    if mode == 'demo':
        fig, ax = plt.subplots(figsize=(10, 8))
        p, v, d = demo_data()
        specs = [(ax, 'Движение и самая удалённая пара', p, v, d)]
        fig.suptitle('Лабораторная 5 • Джарвис и диаметр множества', fontsize=16, y=.98)
        fig.subplots_adjust(top=.84, bottom=.13, left=.1, right=.94)
    else:
        fig, axes = plt.subplots(2, 3, figsize=(14, 9))
        specs = [(ax, *case) for ax, case in zip(axes.flat, BOUNDARY_CASES)]
        fig.suptitle('Диаметр множества • Граничные случаи', fontsize=16, y=.98)
        fig.subplots_adjust(top=.83, bottom=.11, hspace=.65, wspace=.3)
    panels = []
    for ax, title, p, v, d in specs:
        history = simulate(p, v, d, frames)
        ax.set_title(title, pad=57, fontsize=11)
        if len(history[0].positions):
            all_positions = np.concatenate([s.positions for s in history])
            low, high = all_positions.min(axis=0), all_positions.max(axis=0)
            # Поле показа квадратное и рассчитано по всей истории;
            # границы рисунка не участвуют в движении.
            center = (low + high) / 2
            half = max(float(np.max(high - low)) / 2 + .5, d / 2 + .5, 1.)
            ax.set_xlim(center[0] - half, center[0] + half)
            ax.set_ylim(center[1] - half, center[1] + half)
        else:
            ax.set_xlim(-2, 2)
            ax.set_ylim(-2, 2)
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.grid(alpha=.2)
        dots = ax.scatter([], [], s=38, zorder=5)
        hull_line, = ax.plot([], [], '-', color='#577caa', lw=1.5, zorder=2)
        pair_line, = ax.plot([], [], '-o', color='#d48119', lw=2.5, ms=7, zorder=6)
        fill = Polygon(np.zeros((3, 2)), facecolor='#eaf1fa', edgecolor='none', zorder=1)
        fill.set_visible(False)
        ax.add_patch(fill)
        initial = history[0]
        arrows = ax.quiver(initial.positions[:, 0], initial.positions[:, 1],
                           initial.velocities[:, 0], initial.velocities[:, 1],
                           angles='xy', scale_units='xy', scale=.25, width=.005,
                           color='#27799a', zorder=4)
        labels = [ax.text(0, 0, str(i), fontsize=8, color='#46566b') for i in range(len(p))]
        trails = [ax.plot([], [], color='#8fb5c5', alpha=.4, lw=1)[0] for _ in p]
        status = ax.text(0, 1.025, '', transform=ax.transAxes, fontsize=9, va='bottom')
        panels.append((history, d, dots, hull_line, pair_line, fill, arrows, labels, trails, status))
    fig.text(.5, .025, 'Синий: оболочка Джарвиса   •   Оранжевый: диаметр   •   Красный: скорости пары изменены на −V',
             ha='center', fontsize=10)

    def update(frame):
        artists = []
        for history, d, dots, hull_line, pair_line, fill, arrows, labels, trails, status in panels:
            snap = history[frame]
            p = snap.positions
            colors = ['#27799a'] * len(p)
            if snap.reversed_pair:
                for i in snap.pair:
                    colors[i] = '#d64d4b'
            dots.set_offsets(p)
            dots.set_color(colors)
            q = p[snap.hull]
            closed = np.vstack([q, q[0]]) if len(q) >= 3 else q
            hull_line.set_data(closed[:, 0], closed[:, 1])
            fill.set_visible(len(q) >= 3)
            if len(q) >= 3:
                fill.set_xy(q)
            pair_points = p[list(snap.pair)] if snap.pair is not None else np.empty((0, 2))
            pair_line.set_data(pair_points[:, 0], pair_points[:, 1])
            pair_line.set_color('#d64d4b' if snap.reversed_pair else '#d48119')
            arrows.set_offsets(p)
            arrows.set_UVC(snap.velocities[:, 0], snap.velocities[:, 1])
            groups = {}
            for i, position in enumerate(p):
                groups.setdefault(tuple(position), []).append(i)
            for members in groups.values():
                for i in members:
                    labels[i].set_visible(i == members[0])
                label = labels[members[0]]
                label.set_text(','.join(map(str, members)))
                label.set_ha('right' if len(members) > 1 else 'left')
                offset = np.array([-.08, .08]) if len(members) > 1 else np.array([.08, .08])
                label.set_position(p[members[0]] + offset)
            for i, trail in enumerate(trails):
                past = np.array([state.positions[i] for state in history[max(0, frame - 20):frame + 1]])
                trail.set_data(past[:, 0], past[:, 1])
            pair = 'нет двух точек' if snap.pair is None else f'{snap.pair[0]}, {snap.pair[1]}'
            event = 'V → −V у пары' if snap.reversed_pair else 'скорости сохранены'
            status.set_text(f'Кадр {frame}  |  D={d:g}  |  диаметр={snap.diameter:.4f}\n'
                            f'Самая удалённая пара: {pair}\n{event}')
            artists.extend([dots, hull_line, pair_line, fill, arrows, status, *labels, *trails])
        return artists

    animation = FuncAnimation(fig, update, frames=frames, interval=interval,
                              repeat=True, blit=False, cache_frame_data=False)
    update(0)
    return fig, animation, update


def run_tests():
    checks = []
    def check(name, condition):
        if not condition:
            raise AssertionError(name)
        checks.append(name)

    cases = [
        ('Пустое множество', [], [], 0., None),
        ('Одна точка', [[1, 2]], [0], 0., None),
        ('Две точки', [[0, 0], [3, 4]], [0, 1], 25., (0, 1)),
        ('Совпадающие точки', [[1, 2], [1, 2], [1, 2]], [0], 0., (0, 1)),
        ('Коллинеарные точки', [[2, 2], [0, 0], [4, 4], [3, 3]], [1, 2], 32., (1, 2)),
        ('Квадрат: равные диаметры', [[0, 0], [4, 0], [4, 4], [0, 4]],
         [0, 1, 2, 3], 32., (0, 2)),
        ('Повторы на оболочке', [[0, 0], [0, 0], [4, 0], [4, 4], [0, 4], [2, 2]],
         [0, 2, 3, 4], 32., (0, 3)),
        ('Почти коллинеарные точки', [[0, 0], [1, 1e-10], [2, 0]], [0, 2, 1], 4., (0, 2)),
    ]
    for name, points, expected_hull, expected_d, expected_pair in cases:
        hull = jarvis_hull(points)
        diameter, pair = set_diameter(points, hull)
        check(name + ': Джарвис', hull.tolist() == expected_hull)
        check(name + ': диаметр/пара', diameter == expected_d and pair == expected_pair)

    def brute_force(p):
        best, pair = 0., None
        for i in range(len(p)):
            for j in range(i + 1, len(p)):
                d = float(np.dot(p[i] - p[j], p[i] - p[j]))
                if pair is None or d > best:
                    best, pair = d, (i, j)
        return best, pair

    rng = np.random.default_rng(17)
    for trial in range(500):
        count = int(rng.integers(0, 65))
        p = (rng.integers(-10, 11, (count, 2)).astype(float) if trial % 2
             else rng.normal(size=(count, 2)))
        hull = jarvis_hull(p)
        best, pair = set_diameter(p, hull)
        expected, expected_pair = brute_force(p)
        assert best == expected and pair == expected_pair, (trial, pair, expected_pair)
        if len(hull) >= 3:
            assert all(turn(p[hull[i - 2]], p[hull[i - 1]], p[hull[i]]) > 0 for i in range(len(hull)))
            for a, b in zip(hull, np.roll(hull, -1)):
                assert all(turn(p[a], p[b], s) >= -1e-10 for s in p)
    check('500 наборов: калиперы совпадают с полным перебором, оболочки выпуклы и охватывают точки', True)
    history = simulate([[0, 0], [2, 0]], [[0, 0], [0, 0]], 2., 4)
    check('Диаметр точно D: нет разворота', not any(s.reversed_pair for s in history))
    _, p, v, d = BOUNDARY_CASES[2]
    history = simulate(p, v, d, 72)
    check('Движение на расстоянии точно D: нет разворота на всех кадрах',
          all(s.diameter == d and not s.reversed_pair for s in history))
    for offset, expected in [(-1e-8, False), (1e-8, True)]:
        history = simulate([[0., 0.], [2. + offset, 0.]], [[0., 0.], [0., 0.]], 2., 1)
        check(f'Диаметр D{offset:+g}: строгое сравнение', history[0].reversed_pair == expected)
    history = simulate([[0, 0], [3, 0]], [[-.1, 0], [.1, 0]], 2., 4)
    check('Диаметр больше D: скорость пары меняет знак', np.allclose(history[0].velocities, [[.1, 0], [-.1, 0]]))
    check('Условие проверяется на каждом кадре', all(s.reversed_pair for s in history))
    history = simulate([[0, 0], [2, 0], [1, 0]], [[-.1, 0], [.1, 0], [.03, .02]], 1., 3)
    check('Скорость третьей точки сохраняется', all(np.allclose(s.velocities[2], [.03, .02]) for s in history))
    history = simulate([[0, 0], [1.8, 0]], [[-.2, 0], [.2, 0]], 2., 2)
    check('Пересечение порога D после шага', not history[0].reversed_pair and history[1].reversed_pair)
    check('Позиции не корректируются при развороте', np.allclose(history[1].positions, [[-.2, 0], [2., 0]]))
    p, v, d = demo_data()
    history = simulate(p, v, d, 180)
    check('Основная демонстрация содержит развороты', any(s.reversed_pair for s in history))
    check('Модули всех скоростей постоянны',
          all(np.allclose(np.linalg.norm(s.velocities, axis=1), np.linalg.norm(v, axis=1)) for s in history))
    for i in range(1, len(history)):
        prev, current = history[i - 1], history[i]
        assert np.allclose(current.positions, prev.positions + prev.velocities)
        expected = prev.velocities.copy()
        if current.reversed_pair:
            expected[list(current.pair)] *= -1
        assert np.allclose(current.velocities, expected)
    check('Каждый шаг и каждый разворот соответствуют модели', True)
    for invalid in [-1., np.nan, np.inf]:
        try:
            simulate([], [], invalid)
        except ValueError:
            pass
        else:
            raise AssertionError('Некорректный D')
    check('Некорректные D отклоняются', True)
    for name in checks:
        print('OK:', name)
    print(f'Итого: {len(checks)} проверок, все пройдены.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['demo', 'boundary'], default='demo')
    parser.add_argument('--frames', type=int, default=160)
    parser.add_argument('--interval', type=int, default=80, help='мс между кадрами')
    parser.add_argument('--save', type=Path, help='GIF или MP4 (MP4 требует ffmpeg)')
    parser.add_argument('--snapshot', type=Path, help='PNG последнего кадра')
    parser.add_argument('--test', action='store_true')
    args = parser.parse_args()
    if args.test:
        run_tests()
        return
    if args.frames < 1 or args.interval <= 0:
        parser.error('--frames >= 1, --interval > 0')
    if args.save or args.snapshot:
        import matplotlib
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, anim, update = make_animation(args.mode, args.frames, args.interval)
    if args.save:
        suffix = args.save.suffix.lower()
        if suffix not in ('.gif', '.mp4'):
            parser.error('--save поддерживает GIF или MP4')
        anim.save(args.save, writer='pillow' if suffix == '.gif' else 'ffmpeg',
                  fps=1000 / args.interval, dpi=100)
        print(f'Сохранено: {args.save}')
    if args.snapshot:
        update(args.frames - 1)
        fig.savefig(args.snapshot, dpi=100)
    if not args.save and not args.snapshot:
        plt.show()
    else:
        plt.close(fig)


if __name__ == '__main__':
    main()
