#!/usr/bin/env python3
"""Лабораторная 4: выпуклая оболочка алгоритмом Грэхема.

python graham_lab.py
python graham_lab.py --mode boundary
python graham_lab.py --save graham_demo.gif
python graham_lab.py --test

Каждое добавление или удаление элемента стека сохраняется отдельно.
Промежуточный результат — открытая ломаная по стеку. Итоговая оболочка
замыкается при наличии хотя бы трёх вершин. Коллинеарные промежуточные
точки не включаются в минимальный список вершин оболочки.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import numpy as np


def turn(a, b, c):
    """Ориентированная удвоенная площадь: >0 — левый поворот."""
    u, v = b - a, c - a
    return float(u[0] * v[1] - u[1] * v[0])


def normalize(points):
    """Удалить точные повторы; координаты должны быть конечными."""
    p = np.asarray(points, dtype=float)
    if p.size == 0:
        return np.empty((0, 2), dtype=float)
    if p.ndim != 2 or p.shape[1] != 2 or not np.isfinite(p).all():
        raise ValueError('Ожидается конечный массив точек размера (n, 2)')
    return np.unique(p, axis=0)


def polar_order(points):
    """Начало: минимальный y, затем x. Сортировка по углу и расстоянию."""
    p = normalize(points)
    if len(p) < 2:
        return p
    index = int(np.lexsort((p[:, 0], p[:, 1]))[0])
    anchor = p[index]
    rest = np.delete(p, index, axis=0)
    delta = rest - anchor
    angles = np.arctan2(delta[:, 1], delta[:, 0])
    distances = np.einsum('ij,ij->i', delta, delta)
    order = np.lexsort((distances, angles))
    return np.vstack([anchor, rest[order]])


@dataclass(frozen=True)
class Frame:
    stack: tuple[int, ...]
    operation: str
    candidate: int | None
    removed: int | None
    message: str


def graham_scan(points):
    """Вернуть (полярный порядок, кадры, оболочка).

    Сортировка O(n log n), обработка стека O(n), рабочий стек O(n).
    История для анимации содержит копии стека и требует до O(n²) памяти.
    Вершины оболочки перечислены против часовой стрелки.
    """
    ordered = polar_order(points)
    stack = []
    frames = [Frame((), 'start', None, None, 'Начало: стек пуст')]
    for index, point in enumerate(ordered):
        while len(stack) >= 2:
            value = turn(ordered[stack[-2]], ordered[stack[-1]], point)
            if value > 0:
                break
            removed = stack.pop()
            reason = 'правый поворот' if value < 0 else 'коллинеарность'
            frames.append(Frame(tuple(stack), 'pop', index, removed,
                                f'POP #{removed}: {reason}'))
        stack.append(index)
        frames.append(Frame(tuple(stack), 'push', index, None,
                            f'PUSH #{index}: добавить точку'))
    hull = ordered[stack] if stack else np.empty((0, 2), dtype=float)
    return ordered, frames, hull


def demo_points(seed=19):
    rng = np.random.default_rng(seed)
    return np.vstack([rng.uniform([1., 1.], [9., 8.], (19, 2)),
                      [[0., 0.], [3., 0.], [6., 0.], [9., 0.],
                       [10., 4.], [7., 9.], [2., 9.], [0., 5.]]])


BOUNDARY_CASES = [
    ('1. Пустое множество', []),
    ('2. Одна точка', [[2, 2]]),
    ('3. Две разные точки', [[0, 0], [4, 3]]),
    ('4. Все точки коллинеарны', [[0, 0], [1, 1], [2, 2], [3, 3], [4, 4]]),
    ('5. Повторы и внутренняя точка',
     [[0, 0], [4, 0], [4, 4], [0, 4], [2, 2], [0, 0], [4, 4], [2, 2]]),
    ('6. Несколько точек на одном луче',
     [[0, 0], [1, 0], [2, 0], [4, 0], [1, 1], [2, 2], [4, 4], [0, 4]]),
]


def make_animation(mode='demo', interval=600):
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from matplotlib.patches import Polygon
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10})
    if mode == 'demo':
        fig, ax = plt.subplots(figsize=(10, 8))
        specifications = [(ax, 'Алгоритм Грэхема', demo_points())]
        fig.subplots_adjust(top=.84, bottom=.16, left=.1, right=.94)
        fig.suptitle('Лабораторная 4 • Построение выпуклой оболочки', fontsize=16, y=.975)
    else:
        fig, axes = plt.subplots(2, 3, figsize=(14, 9))
        specifications = [(ax, title, points) for ax, (title, points)
                          in zip(axes.flat, BOUNDARY_CASES)]
        fig.subplots_adjust(top=.85, bottom=.13, hspace=.65, wspace=.3)
        fig.suptitle('Алгоритм Грэхема • Граничные случаи', fontsize=16, y=.98)
    panels = []
    for ax, title, points in specifications:
        raw = np.asarray(points, dtype=float).reshape(-1, 2)
        ordered, frames, hull = graham_scan(points)
        ax.set_title(title, pad=43, fontsize=11)
        if len(raw):
            ax.scatter(raw[:, 0], raw[:, 1], color='#8794a2', s=32, zorder=3)
            low, high = raw.min(axis=0), raw.max(axis=0)
            padding = max(float(np.max(high - low)) * .12, .8)
            ax.set_xlim(low[0] - padding, high[0] + padding)
            ax.set_ylim(low[1] - padding, high[1] + padding)
        else:
            ax.set_xlim(-1, 5)
            ax.set_ylim(-1, 5)
        ax.set_aspect('equal', adjustable='box')
        ax.grid(alpha=.18)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        for i, s in enumerate(ordered):
            nearby = any(np.linalg.norm(s - previous) < .3 for previous in ordered[:i])
            offset = (-6, -11) if nearby else (5, 5)
            ax.annotate(str(i), s, xytext=offset, textcoords='offset points',
                        ha='right' if nearby else 'left',
                        fontsize=9 if mode == 'demo' else 8, color='#596777')
        line, = ax.plot([], [], '-o', color='#2368b5', lw=2, ms=5, zorder=4)
        candidate = ax.scatter([], [], s=120, facecolors='none', edgecolors='#df9019',
                               lw=2, zorder=6)
        removed, = ax.plot([], [], 'x', color='#d74b45', ms=10, mew=2, zorder=7)
        preview, = ax.plot([], [], '--', color='#df9019', alpha=.65, zorder=2)
        fill = Polygon(np.zeros((3, 2)), facecolor='#dcece3', alpha=.7, edgecolor='none', zorder=1)
        fill.set_visible(False)
        ax.add_patch(fill)
        header = ax.text(0, 1.025, '', transform=ax.transAxes, va='bottom', fontsize=9)
        stack_text = ax.text(0, -.13 if mode == 'demo' else -.2, '', transform=ax.transAxes, va='top', fontsize=9,
                             color='#2368b5')
        panels.append((ordered, frames, hull, line, candidate, removed, preview,
                       fill, header, stack_text))
    total = max(len(panel[1]) for panel in panels)
    fig.text(.5, .025, 'Синий: стек   •   Оранжевый: рассматриваемая точка   •   Красный ×: удалена из стека',
             ha='center', fontsize=10)

    def update(number):
        artists = []
        for ordered, frames, hull, line, candidate, removed, preview, fill, header, text in panels:
            index = min(number, len(frames) - 1)
            frame = frames[index]
            done = index == len(frames) - 1
            stack = ordered[list(frame.stack)] if frame.stack else np.empty((0, 2))
            display = np.vstack([stack, stack[0]]) if done and len(stack) >= 3 else stack
            line.set_data(display[:, 0], display[:, 1])
            line.set_color('#24845b' if done else '#2368b5')
            candidate.set_offsets(ordered[[frame.candidate]] if frame.candidate is not None
                                  and not done else np.empty((0, 2)))
            if frame.removed is None:
                removed.set_data([], [])
            else:
                removed.set_data([ordered[frame.removed, 0]], [ordered[frame.removed, 1]])
            if frame.operation == 'pop' and len(stack):
                edge = np.vstack([stack[-1], ordered[frame.candidate]])
                preview.set_data(edge[:, 0], edge[:, 1])
            else:
                preview.set_data([], [])
            fill.set_visible(done and len(hull) >= 3)
            if done and len(hull) >= 3:
                fill.set_xy(hull)
            prefix = f'Изменение стека {index}/{len(frames) - 1}'
            result = f'Готово: вершин оболочки {len(hull)}' if done else frame.message
            header.set_text(prefix + '\n' + result)
            values = ', '.join(map(str, frame.stack))
            text.set_text('Стек: [' + values + ']')
            artists.extend([line, candidate, removed, preview, fill, header, text])
        return artists

    animation = FuncAnimation(fig, update, frames=total, interval=interval,
                              repeat=True, repeat_delay=1800, blit=False, cache_frame_data=False)
    update(0)
    return fig, animation, update, total


def run_tests():
    checks = []
    def check(name, condition):
        if not condition:
            raise AssertionError(name)
        checks.append(name)

    cases = [
        ('Пустое множество', [], []),
        ('Одна точка', [[2, 2]], [[2, 2]]),
        ('Две точки', [[4, 3], [0, 0]], [[0, 0], [4, 3]]),
        ('Только повторы', [[2, 2]] * 5, [[2, 2]]),
        ('Коллинеарные точки', [[4, 4], [2, 2], [0, 0], [3, 3]], [[0, 0], [4, 4]]),
        ('Горизонтальная прямая', [[2, 1], [-2, 1], [4, 1]], [[-2, 1], [4, 1]]),
        ('Вертикальная прямая', [[2, 1], [2, -3], [2, 4]], [[2, -3], [2, 4]]),
        ('Квадрат с внутренней точкой', [[0, 0], [4, 0], [4, 4], [0, 4], [2, 2]],
         [[0, 0], [4, 0], [4, 4], [0, 4]]),
        ('Точки на сторонах', [[0, 0], [2, 0], [4, 0], [4, 2], [4, 4], [0, 4], [0, 2]],
         [[0, 0], [4, 0], [4, 4], [0, 4]]),
        ('Один луч и повторы', BOUNDARY_CASES[-1][1], [[0, 0], [4, 0], [4, 4], [0, 4]]),
        ('Почти коллинеарный треугольник', [[0, 0], [1, 1e-10], [2, 0]],
         [[0, 0], [2, 0], [1, 1e-10]]),
    ]
    for name, points, expected in cases:
        ordered, frames, hull = graham_scan(points)
        check(name, np.array_equal(hull, np.asarray(expected, float).reshape(-1, 2)))
        for previous, current in zip(frames, frames[1:]):
            if current.operation == 'push':
                assert current.stack == previous.stack + (current.candidate,)
            elif current.operation == 'pop':
                assert current.stack == previous.stack[:-1]
                assert current.removed == previous.stack[-1]
            else:
                raise AssertionError('Кадр без изменения стека')
        check(name + ': один кадр на push/pop', True)
        check(name + ': последняя ломаная совпадает с оболочкой',
              np.array_equal(ordered[list(frames[-1].stack)], hull))

    def reference(points):
        # Независимая эталонная проверка цепями, только в тестах.
        p = normalize(points)
        if len(p) < 2:
            return p
        lower, upper = [], []
        for source, chain in [(p, lower), (p[::-1], upper)]:
            for s in source:
                while len(chain) >= 2 and turn(chain[-2], chain[-1], s) <= 0:
                    chain.pop()
                chain.append(s)
        return np.array(lower[:-1] + upper[:-1])

    rng = np.random.default_rng(23)
    for trial in range(300):
        count = int(rng.integers(0, 80))
        points = (rng.integers(-15, 16, size=(count, 2)).astype(float) if trial % 2
                  else rng.normal(size=(count, 2)))
        _, _, hull = graham_scan(points)
        expected = reference(points)
        assert set(map(tuple, hull)) == set(map(tuple, expected))
        if len(hull) >= 3:
            assert all(turn(hull[i - 2], hull[i - 1], hull[i]) > 0 for i in range(len(hull)))
            for a, b in zip(hull, np.roll(hull, -1, axis=0)):
                assert all(turn(a, b, s) >= -1e-10 for s in points)
    check('300 наборов: эталонная оболочка, выпуклость и охват исходных точек', True)
    for invalid in [[[0, np.nan]], [[0, np.inf]], [1, 2, 3]]:
        try:
            graham_scan(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError('Некорректные данные приняты')
    check('Некорректные координаты отклоняются', True)
    check('Общая демонстрация содержит POP',
          any(f.operation == 'pop' for f in graham_scan(demo_points())[1]))
    for name in checks:
        print('OK:', name)
    print(f'Итого: {len(checks)} проверок, все пройдены.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['demo', 'boundary'], default='demo')
    parser.add_argument('--interval', type=int, default=600, help='мс между изменениями стека')
    parser.add_argument('--save', type=Path, help='GIF или MP4 (MP4 требует ffmpeg)')
    parser.add_argument('--snapshot', type=Path, help='PNG последнего кадра')
    parser.add_argument('--test', action='store_true')
    args = parser.parse_args()
    if args.test:
        run_tests()
        return
    if args.interval <= 0:
        parser.error('--interval должен быть положительным')
    if args.save or args.snapshot:
        import matplotlib
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, anim, update, count = make_animation(args.mode, args.interval)
    if args.save:
        if args.save.suffix.lower() not in ('.gif', '.mp4'):
            parser.error('--save поддерживает GIF и MP4')
        anim.save(args.save, writer='pillow' if args.save.suffix.lower() == '.gif' else 'ffmpeg',
                  fps=1000 / args.interval, dpi=100)
        print(f'Сохранено: {args.save}; кадров: {count}')
    if args.snapshot:
        update(count - 1)
        fig.savefig(args.snapshot, dpi=100)
    if not args.save and not args.snapshot:
        plt.show()
    else:
        plt.close(fig)


if __name__ == '__main__':
    main()
