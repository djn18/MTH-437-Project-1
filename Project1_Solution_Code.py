"""Reference computations for MTH 437 Project 1.

Run this file with Python 3. It prints representative numerical results and
creates Project1_Newton_Convergence.png in the same directory.
"""

from math import cos, log10, sin, sqrt
from pathlib import Path


def bisection(f, a, b, tolerance, max_iterations=1000):
    """Return a midpoint approximation and the complete iteration history."""
    fa = f(a)
    fb = f(b)
    if fa == 0.0:
        return a, [(0, a, a, a, 0.0)]
    if fb == 0.0:
        return b, [(0, b, b, b, 0.0)]
    if fa * fb > 0.0:
        raise ValueError("The endpoints do not bracket a sign change.")

    history = []
    for n in range(max_iterations + 1):
        midpoint = (a + b) / 2.0
        bound = (b - a) / 2.0
        fm = f(midpoint)
        history.append((n, a, b, midpoint, bound))
        if fm == 0.0 or bound <= tolerance:
            return midpoint, history
        if fa * fm < 0.0:
            b = midpoint
        else:
            a = midpoint
            fa = fm
    raise RuntimeError("Bisection reached the maximum iteration count.")


def newton(f, derivative, x0, tolerance, max_iterations=100):
    """Newton iteration using the relative-step stopping criterion."""
    history = [(0, x0, None)]
    x = x0
    for n in range(1, max_iterations + 1):
        dfx = derivative(x)
        if dfx == 0.0:
            raise ZeroDivisionError("Newton derivative is zero.")
        x_new = x - f(x) / dfx
        eta = abs(x_new - x) / max(1.0, abs(x_new))
        history.append((n, x_new, eta))
        if eta <= tolerance:
            return x_new, history
        x = x_new
    raise RuntimeError("Newton iteration reached its iteration limit.")


def secant(f, x0, x1, tolerance, max_iterations=100):
    """Secant iteration using the relative-step stopping criterion."""
    history = [(0, x0, None), (1, x1, abs(x1 - x0) / max(1.0, abs(x1)))]
    f0 = f(x0)
    f1 = f(x1)
    for n in range(2, max_iterations + 1):
        denominator = f1 - f0
        if denominator == 0.0:
            raise ZeroDivisionError("The secant denominator is zero.")
        x2 = x1 - f1 * (x1 - x0) / denominator
        eta = abs(x2 - x1) / max(1.0, abs(x2))
        history.append((n, x2, eta))
        if eta <= tolerance:
            return x2, history
        x0, x1 = x1, x2
        f0, f1 = f1, f(x1)
    raise RuntimeError("Secant iteration reached its iteration limit.")


def sin_minus_x_stable(x):
    """Evaluate sin(x)-x without cancellation when x is small."""
    if abs(x) >= 0.1:
        return sin(x) - x
    x2 = x * x
    factor = 1.0 - x2 / 20.0 + x2**2 / 840.0
    factor -= x2**3 / 60480.0
    factor += x2**4 / 6652800.0
    return -(x * x2 / 6.0) * factor


def cos_minus_one_stable(x):
    """Evaluate cos(x)-1 by an equivalent cancellation-free identity."""
    return -2.0 * sin(x / 2.0) ** 2


def multiple_root_history(stable, max_iterations=100):
    """Return Newton iterates for sin(x)-x, including stagnation information."""
    x = 0.5
    history = [(0, x)]
    for _ in range(max_iterations):
        if abs(x) <= 1.0e-10:
            return history, "tolerance reached"
        if stable:
            numerator = sin_minus_x_stable(x)
            denominator = cos_minus_one_stable(x)
        else:
            numerator = sin(x) - x
            denominator = cos(x) - 1.0
        if denominator == 0.0:
            return history, "zero computed derivative"
        x_new = x - numerator / denominator
        if x_new == x:
            return history, "stagnation"
        x = x_new
        history.append((len(history), x))
    return history, "maximum iteration count reached"


def make_plot(direct_history, stable_history, output_path):
    """Create a dependency-light semilog plot with Pillow."""
    from PIL import Image, ImageDraw, ImageFont

    width, height = 1600, 950
    left, right, top, bottom = 150, 70, 150, 135
    plot_width = width - left - right
    plot_height = height - top - bottom
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font_path = "/System/Library/Fonts/Supplemental/Arial.ttf"
    bold_path = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
    font = ImageFont.truetype(font_path, 28)
    small = ImageFont.truetype(font_path, 24)
    bold = ImageFont.truetype(bold_path, 31)

    x_min, x_max = 0.0, 56.0
    y_min, y_max = -11.0, 0.0

    def point(n, error):
        px = left + (n - x_min) * plot_width / (x_max - x_min)
        log_error = max(y_min, min(y_max, log10(error)))
        py = top + (y_max - log_error) * plot_height / (y_max - y_min)
        return px, py

    for power in range(-10, 1, 2):
        y = point(0, 10.0**power)[1]
        draw.line((left, y, width - right, y), fill=(215, 215, 215), width=2)
        draw.text((40, y - 14), f"10^{power}", fill="black", font=small)
    for n in range(0, 57, 10):
        x = point(n, 1.0)[0]
        draw.line((x, top, x, height - bottom), fill=(235, 235, 235), width=2)
        draw.text((x - 15, height - bottom + 18), str(n), fill="black", font=small)

    draw.line((left, top, left, height - bottom), fill="black", width=4)
    draw.line((left, height - bottom, width - right, height - bottom), fill="black", width=4)
    draw.text((width / 2 - 110, height - 65), "Iteration n", fill="black", font=font)
    draw.text((25, 92), "Absolute error |x_n| (log scale)", fill="black", font=font)
    draw.text((430, 28), "Newton convergence for sin(x) - x", fill="black", font=bold)

    direct_for_plot = list(direct_history)
    last_n, last_x = direct_for_plot[-1]
    direct_for_plot.extend((n, last_x) for n in range(last_n + 1, 57))
    direct_points = [point(n, abs(x)) for n, x in direct_for_plot]
    stable_points = [point(n, abs(x)) for n, x in stable_history]
    ideal_points = [point(n, 0.5 * (2.0 / 3.0) ** n) for n in range(57)]

    draw.line(stable_points, fill=(0, 86, 150), width=6)
    draw.line(direct_points, fill=(213, 94, 0), width=6)
    for px, py in direct_points[::4]:
        draw.ellipse((px - 7, py - 7, px + 7, py + 7), outline=(213, 94, 0), width=4)
    for k in range(0, len(ideal_points) - 1, 2):
        draw.line((ideal_points[k], ideal_points[k + 1]), fill=(20, 20, 20), width=4)

    legend_y = top + 30
    legend_x = width - right - 570
    draw.rectangle((legend_x - 25, legend_y - 20, width - right, legend_y + 150),
                   fill="white", outline=(100, 100, 100), width=2)
    entries = [
        ((0, 86, 150), "Stable evaluation", "solid"),
        ((213, 94, 0), "Direct evaluation (stagnates)", "solid"),
        ((20, 20, 20), "Reference: 0.5(2/3)^n", "dashed"),
    ]
    for index, (color, label, style) in enumerate(entries):
        y = legend_y + index * 52
        if style == "dashed":
            for x in range(legend_x, legend_x + 120, 30):
                draw.line((x, y, x + 18, y), fill=color, width=5)
        else:
            draw.line((legend_x, y, legend_x + 120, y), fill=color, width=6)
        draw.text((legend_x + 145, y - 16), label, fill="black", font=small)

    image.save(output_path, dpi=(180, 180))


def main():
    cube_root_two = 2.0 ** (1.0 / 3.0)
    approximation, history = bisection(lambda x: x**3 - 2.0, 1.0, 2.0, 1.0e-8)
    print("Problem 1 manufactured root")
    print(approximation, abs(approximation - cube_root_two), history[-1][4])

    cube_root_six = 6.0 ** (1.0 / 3.0)
    _, bracket_history = bisection(lambda x: x**3 - 6.0, 1.0, 2.0, 1.0e-3)
    x0 = bracket_history[-1][3]
    approximation, history = newton(
        lambda x: x**3 - 6.0, lambda x: 3.0 * x**2, x0, 1.0e-12
    )
    print("Problem 2 Newton root")
    print(approximation, abs(approximation - cube_root_six), len(history) - 1)

    _, bracket_history = bisection(lambda x: x**2 - 11.0, 3.0, 4.0, 1.0e-3)
    a, b = bracket_history[-1][1], bracket_history[-1][2]
    approximation, history = secant(lambda x: x**2 - 11.0, a, b, 1.0e-10)
    print("Problem 2 secant root")
    print(approximation, abs(approximation - sqrt(11.0)), len(history) - 1)

    direct_history, direct_status = multiple_root_history(stable=False)
    stable_history, stable_status = multiple_root_history(stable=True)
    print("Problem 3 direct:", direct_status, direct_history[-1])
    print("Problem 3 stable:", stable_status, stable_history[-1])
    output = Path(__file__).with_name("Project1_Newton_Convergence.png")
    make_plot(direct_history, stable_history, output)
    print("Wrote", output)


if __name__ == "__main__":
    main()
