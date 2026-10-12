#!/usr/bin/env python3
"""Лабораторная 3. Python 3.10+, NumPy, Matplotlib, Pillow.

python polygon_lab.py                         # интерактивная общая анимация
python polygon_lab.py --mode boundary         # граничные случаи
python polygon_lab.py --save demo.gif         # сохранить GIF
python polygon_lab.py --test                  # проверки алгоритмов

Покадровая модель буквально из условия: кандидат = S + V. Если он вне Q,
отражение от первой пересечённой стороны меняет V, а S остаётся текущим.
По уточнению пользователя точка останавливается точно на границе P:
проверяется весь отрезок перемещения и выбирается первое пересечение.
Выход через вершину Q отмечается UNDEFINED: правило отражения не задано.
"""

from __future__ import annotations
import argparse
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
import numpy as np

EPS = 1e-9


class Location(IntEnum):
    OUTSIDE = -1
    BOUNDARY = 0
    INSIDE = 1


NAMES = {Location.OUTSIDE: 'снаружи', Location.BOUNDARY: 'на границе',
         Location.INSIDE: 'внутри'}


def cross(a, b):
    return float(a[0] * b[1] - a[1] * b[0])


def on_segment(s, a, b):
    """Допуск EPS задан в единицах длины, а не площади."""
    d = b - a
    length = np.linalg.norm(d)
    if length <= EPS:
        return np.linalg.norm(s - a) <= EPS
    return (abs(cross(d, s - a)) <= EPS * length
            and -EPS * length <= np.dot(s - a, d) <= np.dot(d, d) + EPS * length)


def angle_test(s, polygon):
    """Угловой тест для простого P: сумма ориентированных углов.

    Время O(n), память O(1). Граница выделяется до вычисления atan2.
    Невыпуклость и направление обхода не влияют на результат.
    """
    s = np.asarray(s, dtype=float)
    angle = 0.0
    for i, a in enumerate(polygon):
        b = polygon[(i + 1) % len(polygon)]
        if on_segment(s, a, b):
            return Location.BOUNDARY
        u, v = a - s, b - s
        angle += np.arctan2(cross(u, v), np.dot(u, v))
    return Location.INSIDE if abs(angle) > np.pi else Location.OUTSIDE


def prepare_convex(vertices):
    """Разовая подготовка строго выпуклого Q: обход против часовой стрелки.

    Коллинеарные промежуточные вершины удаляются, затем проверяется выпуклость.
    После подготовки binary_test выполняется за O(log m).
    """
    q = np.asarray(vertices, dtype=float).copy()
    if q.ndim != 2 or q.shape[1] != 2 or not np.isfinite(q).all():
        raise ValueError('Q должен быть конечным массивом (m, 2)')
    if len(q) > 1 and np.linalg.norm(q[0] - q[-1]) <= EPS:
        q = q[:-1]
    if len(q) < 3:
        raise ValueError('Q должен иметь хотя бы три вершины')
    if any(np.linalg.norm(q[(i + 1) % len(q)] - q[i]) <= EPS for i in range(len(q))):
        raise ValueError('Повторяющиеся соседние вершины Q')
    area2 = sum(cross(q[i], q[(i + 1) % len(q)]) for i in range(len(q)))
    if abs(area2) <= EPS:
        raise ValueError('Вырожденный Q')
    if area2 < 0:
        q = q[::-1].copy()
    changed = True
    while changed and len(q) > 3:
        changed = False
        for i in range(len(q)):
            if on_segment(q[i], q[i - 1], q[(i + 1) % len(q)]):
                q = np.delete(q, i, axis=0)
                changed = True
                break
    # Проверка всех опорных полуплоскостей за O(m²) только при подготовке.
    for i in range(len(q)):
        d = q[(i + 1) % len(q)] - q[i]
        if any(cross(d, p - q[i]) < -EPS * np.linalg.norm(d) for p in q):
            raise ValueError('Q не является выпуклым многоугольником с упорядоченными вершинами')
    return q


def binary_test(s, q):
    """Бинарный тест в веере треугольников выпуклого Q, O(log m).

    q предварительно обработан prepare_convex. Внутренние диагонали веера
    не являются границей многоугольника.
    """
    s = np.asarray(s, dtype=float)
    a = q[0]
    if np.linalg.norm(s - a) <= EPS:
        return Location.BOUNDARY
    first, last = q[1] - a, q[-1] - a
    left, right = cross(first, s - a), cross(last, s - a)
    e1, e2 = EPS * np.linalg.norm(first), EPS * np.linalg.norm(last)
    if left < -e1 or right > e2:
        return Location.OUTSIDE
    if abs(left) <= e1:
        return Location.BOUNDARY if on_segment(s, a, q[1]) else Location.OUTSIDE
    if abs(right) <= e2:
        return Location.BOUNDARY if on_segment(s, a, q[-1]) else Location.OUTSIDE
    lo, hi = 1, len(q) - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if cross(q[mid] - a, s - a) >= 0:
            lo = mid
        else:
            hi = mid
    b, c = q[lo], q[hi]
    side = cross(c - b, s - b)
    edge_eps = EPS * np.linalg.norm(c - b)
    if side < -edge_eps:
        return Location.OUTSIDE
    if on_segment(s, b, c):
        return Location.BOUNDARY
    return Location.INSIDE


def reflect(v, edge):
    """Формула из задания: V' = 2 (V·e)/(e·e) e − V."""
    return 2.0 * np.dot(v, edge) / np.dot(edge, edge) * edge - v


class UndefinedCase(RuntimeError):
    """Правило для такого события отсутствует в исходном условии."""


def first_boundary_hit(s, v, polygon):
    """Первое пересечение отрезка S + tV, 0 <= t <= 1, с границей P.

    Возвращает точку на стороне P, а не позицию следующего кадра.
    Обрабатывает вершины, касания и движение вдоль стороны; O(n).
    """
    if np.linalg.norm(v) <= EPS:
        return None
    hits = []
    for i, a in enumerate(polygon):
        b = polygon[(i + 1) % len(polygon)]
        edge = b - a
        if on_segment(s, a, b):
            return s.copy()
        denominator = cross(v, edge)
        if abs(denominator) > EPS * np.linalg.norm(v) * np.linalg.norm(edge):
            t = cross(a - s, edge) / denominator
            u = cross(a - s, v) / denominator
            if 0.0 <= t <= 1.0 and 0.0 <= u <= 1.0:
                # Конструируем результат на самой стороне, чтобы не накопить
                # ошибку позиции. u=0 и u=1 дают точную вершину.
                hits.append((t, a + np.clip(u, 0.0, 1.0) * edge))
        elif abs(cross(v, a - s)) <= EPS * np.linalg.norm(v):
            # Коллинеарные отрезки: берём начало их перекрытия.
            vv = np.dot(v, v)
            ta, tb = np.dot(a - s, v) / vv, np.dot(b - s, v) / vv
            lo, hi = max(0.0, min(ta, tb)), min(1.0, max(ta, tb))
            if lo <= hi:
                hit = s + lo * v
                if abs(lo - ta) <= EPS:
                    hit = a.copy()
                elif abs(lo - tb) <= EPS:
                    hit = b.copy()
                hits.append((lo, hit))
    return min(hits, key=lambda pair: pair[0])[1] if hits else None


def exit_edge(s, v, q):
    """Первая сторона, через которую S + tV выходит из Q, 0 <= t <= 1.

    Для отражения нужен только индекс стороны, S не переносится на границу.
    Сканирование сторон O(m); бинарный тест положения остаётся O(log m).
    """
    candidates = []
    for i, a in enumerate(q):
        edge = q[(i + 1) % len(q)] - a
        rate = cross(edge, v)
        if rate < -EPS * np.linalg.norm(edge):
            t = -cross(edge, s - a) / rate
            if -EPS <= t <= 1 + EPS:
                candidates.append((max(0.0, t), i))
    if not candidates:
        raise ArithmeticError('Не найдена пересекаемая сторона Q')
    t, i = min(candidates)
    hit = s + t * v
    if any(np.linalg.norm(hit - vertex) <= EPS for vertex in q):
        raise UndefinedCase('выход через вершину Q: сторона отражения не задана')
    return i


@dataclass
class Particle:
    position: np.ndarray
    velocity: np.ndarray
    state: str = 'moving'
    event: str = 'движение'


def step(particle, p, q):
    if particle.state != 'moving':
        return
    s, v = particle.position, particle.velocity
    proposed = s + v
    try:
        where = angle_test(s, p)
        if where != Location.OUTSIDE:
            particle.velocity = np.zeros(2)
            particle.state = 'stopped'
            particle.event = 'на границе P: скорость = 0' if where == Location.BOUNDARY else 'внутри P: скорость = 0'
            return
        hit = first_boundary_hit(s, v, p)
        if hit is not None:
            particle.position = hit
            particle.velocity = np.zeros(2)
            particle.state = 'stopped'
            particle.event = 'на границе P: скорость = 0'
            return
        if binary_test(proposed, q) == Location.OUTSIDE:
            i = exit_edge(s, v, q)
            particle.velocity = reflect(v, q[(i + 1) % len(q)] - q[i])
            particle.event = f'отражение от стороны Q: {i + 1}'
            return
        where = angle_test(proposed, p)
        particle.position = proposed
        if where == Location.BOUNDARY:
            particle.velocity = np.zeros(2)
            particle.state = 'stopped'
            particle.event = 'на границе P: скорость = 0'
            return
        if where == Location.INSIDE:
            raise ArithmeticError('Пропущено пересечение границы P')
        else:
            particle.event = 'движение'
    except UndefinedCase as error:
        # Скорость сохраняется: это не физическая остановка по условию.
        # Дальнейшая симуляция этой точки приостановлена до уточнения правила.
        particle.state = 'undefined'
        particle.event = str(error)


DEMO_Q = prepare_convex([(0, 0), (10, 0), (12, 4), (10, 8), (7, 9), (2, 7), (0, 2)])
DEMO_P = np.array([(3, 2), (5, 2), (6, 1), (9, 4), (6, 4), (4, 3), (3, 3)], float)
SQUARE_Q = prepare_convex([(0, 0), (10, 0), (10, 10), (0, 10)])
L_P = np.array([(3, 3), (7, 3), (7, 5), (5, 5), (5, 7), (3, 7)], float)


def trajectory(points, velocities, p, q, frames):
    particles = [Particle(np.array(s, float), np.array(v, float))
                 for s, v in zip(points, velocities)]
    for pt in particles:
        where = angle_test(pt.position, p)
        if binary_test(pt.position, q) == Location.OUTSIDE:
            raise ValueError('Начальная точка вне Q')
        if where == Location.INSIDE:
            pt.state, pt.event = 'stopped', 'начальная точка внутри P'
            pt.velocity = np.zeros(2)
        elif where == Location.BOUNDARY:
            pt.state, pt.event = 'stopped', 'начальная точка на P: скорость = 0'
            pt.velocity = np.zeros(2)
    positions, velocities_out, states, events = [], [], [], []
    for frame in range(frames):
        if frame:
            for pt in particles:
                step(pt, p, q)
        positions.append([pt.position.copy() for pt in particles])
        velocities_out.append([pt.velocity.copy() for pt in particles])
        states.append([pt.state for pt in particles])
        events.append([pt.event for pt in particles])
    return np.array(positions), np.array(velocities_out), np.array(states), events


def demo_points(count=24, seed=31):
    rng = np.random.default_rng(seed)
    points = []
    while len(points) < count:
        s = rng.uniform([0, 0], [12, 9])
        if (binary_test(s, DEMO_Q) == Location.INSIDE
                and angle_test(s, DEMO_P) == Location.OUTSIDE):
            points.append(s)
    # Две управляемые точки гарантируют раннее попадание в P и отражение.
    points[:2] = [np.array([4., 1.]), np.array([9.7, 1.])]
    angles = rng.uniform(0, 2 * np.pi, count)
    velocities = 0.18 * np.column_stack([np.cos(angles), np.sin(angles)])
    velocities[:2] = [[0., .18], [.18, 0.]]
    return points, velocities


def draw_polygons(ax, p, q):
    from matplotlib.patches import Polygon
    ax.add_patch(Polygon(q, facecolor='#eef6ff', edgecolor='#3567bb', lw=2))
    ax.add_patch(Polygon(p, facecolor='#ffe1df', edgecolor='#cf534b', lw=2))
    low, high = q.min(axis=0) - .6, q.max(axis=0) + .6
    ax.set_xlim(low[0], high[0])
    ax.set_ylim(low[1], high[1])
    ax.set_aspect('equal')
    ax.grid(alpha=.18)
    ax.set_xlabel('x')
    ax.set_ylabel('y')


COLORS = {'moving': '#147ca3', 'stopped': '#cf534b', 'undefined': '#8850ad'}


def make_animation(mode='demo', frames=180, interval=60):
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10})
    if mode == 'demo':
        fig, ax = plt.subplots(figsize=(9, 7))
        points, velocities = demo_points()
        scenarios = [(ax, DEMO_P, DEMO_Q, points, velocities, 'Движение точек внутри Q')]
        fig.suptitle('Лабораторная 3 • Угловой и бинарный тесты', fontsize=16, y=.98)
        fig.subplots_adjust(top=.85, bottom=.13)
    else:
        fig, axes = plt.subplots(2, 3, figsize=(14, 9))
        fig.suptitle('Граничные случаи • Результаты покадровой модели', fontsize=16, y=.985)
        fig.subplots_adjust(top=.86, bottom=.09, hspace=.47, wspace=.3)
        specifications = [
            ('1. Выход через сторону Q', [[9., 1.]], [[.35, 0.]]),
            ('2. Шаг внутрь P: стоп на границе', [[2., 4.]], [[.4, 0.]]),
            ('3. Попадание на сторону P', [[2., 4.]], [[.5, 0.]]),
            ('4. Попадание на вершину P', [[2., 2.]], [[.5, .5]]),
            ('5. Выход через вершину Q', [[9., 9.]], [[.6, .6]]),
        ]
        scenarios = [(ax, L_P, SQUARE_Q, pts, vs, title)
                     for ax, (title, pts, vs) in zip(axes.flat, specifications)]
        static_ax = axes.flat[5]
        draw_polygons(static_ax, L_P, SQUARE_Q)
        static_ax.set_title('6. Положение на границе и рядом', pad=44)
        probes = [(np.array([10., 5.]), 'сторона Q'),
                  (np.array([10., 10.]), 'вершина Q'),
                  (np.array([6., 6.]), 'выемка P'),
                  (np.array([5., 6.]), 'сторона P')]
        static_ax.scatter(*np.array([x[0] for x in probes]).T, c='#8850ad', s=34, zorder=5)
        for s, label in probes:
            offset = (8, 10) if label == 'выемка P' else (-5, -13)
            align = 'left' if label == 'выемка P' else 'right'
            static_ax.annotate(label, s, xytext=offset, textcoords='offset points',
                               ha=align, fontsize=8)
        result = [f'{label}: {NAMES[binary_test(s, SQUARE_Q) if "Q" in label else angle_test(s, L_P)]}'
                  for s, label in probes]
        static_ax.text(0, 1.025, '\n'.join(result), transform=static_ax.transAxes,
                       fontsize=8, va='bottom')
    panels = []
    for ax, p, q, pts, vs, title in scenarios:
        draw_polygons(ax, p, q)
        ax.set_title(title, pad=44 if mode == 'boundary' else 38)
        data = trajectory(pts, vs, p, q, frames)
        dots = ax.scatter([], [], s=38, zorder=5)
        trails = [ax.plot([], [], color='#147ca3', alpha=.25, lw=1)[0] for _ in pts]
        arrows = ax.quiver(np.array(pts)[:, 0], np.array(pts)[:, 1],
                           np.array(vs)[:, 0], np.array(vs)[:, 1], color='#147ca3',
                           angles='xy', scale_units='xy', scale=1, width=.005)
        status = ax.text(0, 1.025, '', transform=ax.transAxes, fontsize=9, va='bottom')
        panels.append((data, dots, trails, arrows, status))
    fig.text(.5, .025, 'Синий: движется   •   Красный: остановлен на P, V = 0   •   Фиолетовый: отражение не задано',
             ha='center', fontsize=10)

    def update(frame):
        artists = []
        for (positions, velocities, states, events), dots, trails, arrows, status in panels:
            dots.set_offsets(positions[frame])
            dots.set_color([COLORS[state] for state in states[frame]])
            arrows.set_offsets(positions[frame])
            visible = velocities[frame].copy()
            visible[states[frame] == 'undefined'] = 0
            arrows.set_UVC(visible[:, 0], visible[:, 1])
            for i, trail in enumerate(trails):
                history = positions[max(0, frame - 30):frame + 1, i]
                trail.set_data(history[:, 0], history[:, 1])
            if mode == 'demo':
                moving = np.count_nonzero(states[frame] == 'moving')
                stopped = np.count_nonzero(states[frame] == 'stopped')
                undefined = np.count_nonzero(states[frame] == 'undefined')
                status.set_text(f'Кадр {frame}   |   движутся: {moving}   |   остановлены: {stopped}'
                                f'   |   не определено: {undefined}')
            else:
                velocity = velocities[frame, 0]
                status.set_text(f'Кадр {frame}  |  V=({velocity[0]:.2f}; {velocity[1]:.2f})\n'
                                + events[frame][0])
            artists.extend([dots, arrows, status, *trails])
        return artists

    anim = FuncAnimation(fig, update, frames=frames, interval=interval, blit=False,
                         repeat=True, cache_frame_data=False)
    update(0)
    return fig, anim, update


def run_tests():
    """Аналитические случаи и независимые проверки геометрии/динамики."""
    passed = []
    def check(name, condition):
        if not condition:
            raise AssertionError(name)
        passed.append(name)
    for point, expected in [((4, 4), Location.INSIDE), ((6, 6), Location.OUTSIDE),
                            ((5, 6), Location.BOUNDARY), ((3, 3), Location.BOUNDARY),
                            ((5 - 1e-6, 6), Location.INSIDE),
                            ((5 + 1e-6, 6), Location.OUTSIDE)]:
        check(f'Угловой тест P: {point} → {NAMES[expected]}',
              angle_test(np.array(point), L_P) == expected)
        check(f'Обратный обход P: {point}', angle_test(np.array(point), L_P[::-1]) == expected)
    for point, expected in [((5, 5), Location.INSIDE), ((10, 5), Location.BOUNDARY),
                            ((10, 10), Location.BOUNDARY), ((0, 0), Location.BOUNDARY),
                            ((11, 0), Location.OUTSIDE), ((10 - 1e-6, 5), Location.INSIDE),
                            ((10 + 1e-6, 5), Location.OUTSIDE)]:
        check(f'Бинарный тест Q: {point} → {NAMES[expected]}',
              binary_test(np.array(point), SQUARE_Q) == expected)
    check('Внутренняя диагональ веера Q не является границей',
          binary_test(np.array([3., 3.]), SQUARE_Q) == Location.INSIDE)
    q_collinear = prepare_convex([(0, 0), (5, 0), (10, 0), (10, 10), (0, 10)])
    check('Подготовка удаляет коллинеарную вершину', len(q_collinear) == 4)
    rng = np.random.default_rng(123)
    for q in [SQUARE_Q, DEMO_Q, prepare_convex(DEMO_Q[::-1]), q_collinear]:
        for s in rng.uniform([-2, -2], [14, 12], (1000, 2)):
            # Независимый линейный тест по опорным полуплоскостям.
            inside = all(cross(q[(i+1) % len(q)] - a, s - a) >= 0 for i, a in enumerate(q))
            check_result = binary_test(s, q) == Location.INSIDE
            if check_result != inside:
                raise AssertionError(f'Бинарный тест: {s}')
    check('4000 сравнений бинарного теста с полуплоскостями', True)
    for _ in range(100):
        v, e = rng.normal(size=(2, 2))
        w = reflect(v, e)
        assert np.isclose(np.linalg.norm(v), np.linalg.norm(w))
        assert np.isclose(np.dot(v, e), np.dot(w, e))
        assert np.isclose(cross(e, v), -cross(e, w))
    check('Отражение: модуль, касательная и нормальная компоненты (100 случаев)', True)
    pt = Particle(np.array([9.5, 1.]), np.array([1., .2]))
    step(pt, L_P, SQUARE_Q)
    check('Отражение из текущего S, позиция сохраняется', np.allclose(pt.position, [9.5, 1.]))
    check('Вертикальная сторона: V=(1,.2) → (-1,.2)', np.allclose(pt.velocity, [-1., .2]))
    pt = Particle(np.array([2.8, 4.]), np.array([.4, 0.]))
    step(pt, L_P, SQUARE_Q)
    check('Шаг внутрь P: остановка точно на границе',
          pt.state == 'stopped' and np.all(pt.velocity == 0) and np.allclose(pt.position, [3., 4.]))
    position = pt.position.copy()
    step(pt, L_P, SQUARE_Q)
    check('Остановленная точка остаётся неподвижной', np.array_equal(position, pt.position))
    pt = Particle(np.array([2.5, 4.]), np.array([.5, 0.]))
    step(pt, L_P, SQUARE_Q)
    check('Точное попадание на сторону P: остановка',
          pt.state == 'stopped' and np.all(pt.velocity == 0) and np.allclose(pt.position, [3., 4.]))
    pt = Particle(np.array([9.5, 9.5]), np.array([1., 1.]))
    step(pt, L_P, SQUARE_Q)
    check('Выход через вершину Q выделен как неопределённый', pt.state == 'undefined')
    pt = Particle(np.array([2., 4.]), np.array([6., 0.]))
    step(pt, L_P, SQUARE_Q)
    check('Пересечение P между кадрами: остановка на первом пересечении',
          pt.state == 'stopped' and np.allclose(pt.position, [3., 4.]))
    for name, start, velocity, expected in [
        ('Касание вершины P', [2., 4.], [2., -2.], [3., 3.]),
        ('Шаг через вершину P внутрь', [2., 2.], [2., 2.], [3., 3.]),
        ('Движение вдоль стороны P', [2., 3.], [6., 0.], [3., 3.]),
        ('Начальная точка на P', [3., 4.], [-1., 0.], [3., 4.]),
        ('За P конечная точка вне Q', [2., 4.], [12., 0.], [3., 4.])]:
        pt = Particle(np.array(start), np.array(velocity))
        step(pt, L_P, SQUARE_Q)
        check(name, pt.state == 'stopped' and np.all(pt.velocity == 0)
              and np.allclose(pt.position, expected))
    pts, vs = demo_points()
    positions, velocities, states, _ = trajectory(pts, vs, DEMO_P, DEMO_Q, 240)
    check('Демонстрация: все принятые положения внутри/на Q',
          all(binary_test(s, DEMO_Q) != Location.OUTSIDE for s in positions.reshape(-1, 2)))
    check('Демонстрация содержит остановки', np.any(states == 'stopped'))
    check('Демонстрация не содержит неопределённых событий', not np.any(states == 'undefined'))
    check('Скорости остановленных точек нулевые', np.all(velocities[states == 'stopped'] == 0))
    check('Все остановленные демонстрационные точки находятся на границе P',
          all(angle_test(s, DEMO_P) == Location.BOUNDARY for s in positions[states == 'stopped']))
    print('\n'.join('OK: ' + name for name in passed))
    print(f'Итого: {len(passed)} проверок, все пройдены.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['demo', 'boundary'], default='demo')
    parser.add_argument('--frames', type=int, default=180)
    parser.add_argument('--interval', type=int, default=60, help='мс между кадрами')
    parser.add_argument('--save', type=Path, help='GIF или MP4 (MP4 требует ffmpeg)')
    parser.add_argument('--snapshot', type=Path, help='PNG последнего кадра')
    parser.add_argument('--test', action='store_true')
    args = parser.parse_args()
    if args.test:
        run_tests()
        return
    if args.frames < 2 or args.interval <= 0:
        parser.error('--frames >= 2, --interval > 0')
    if args.save or args.snapshot:
        import matplotlib
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, anim, update = make_animation(args.mode, args.frames, args.interval)
    if args.save:
        suffix = args.save.suffix.lower()
        if suffix not in ('.gif', '.mp4'):
            parser.error('--save поддерживает .gif и .mp4')
        anim.save(args.save, writer='pillow' if suffix == '.gif' else 'ffmpeg',
                  fps=1000 / args.interval, dpi=90)
        print(f'Сохранено: {args.save}')
    if args.snapshot:
        update(args.frames - 1)
        fig.savefig(args.snapshot, dpi=120)
    if not args.save and not args.snapshot:
        plt.show()
    else:
        plt.close(fig)


if __name__ == '__main__':
    main()
