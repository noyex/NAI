import sys
import numpy as np
import matplotlib.pyplot as plt

# Test functions

def himmelblaus_function(x, y):
    return (x**2 + y - 11)**2 + (x + y**2 - 7)**2

def matyas_function(x, y):
    return 0.26 * (x**2 + y**2) - 0.48 * x * y

def booth_function(x, y):
    return (x + 2*y - 7)**2 + (2*x + y - 5)**2

def ackley_function(x, y):
    term1 = -20 * np.exp(-0.2 * np.sqrt(0.5 * (x ** 2 + y ** 2)))
    term2 = -np.exp(0.5 * (np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y)))
    return term1 + term2 + np.e + 20

def holder_table_function(x, y):
    term1 = np.sin(x) * np.cos(y)
    term2 = np.exp(abs(1 - np.sqrt(x ** 2 + y ** 2) / np.pi))
    return -abs(term1 * term2)

functions = {
    "himmelblaus": himmelblaus_function,
    "matyas": matyas_function,
    "booth": booth_function,
    "ackley": ackley_function,
    "holder_table": holder_table_function
}

def plot_function(func, xlim=(-5, 5), ylim=(-5, 5)):
    x = np.linspace(xlim[0], xlim[1], 400)
    y = np.linspace(ylim[0], ylim[1], 400)
    X, Y = np.meshgrid(x, y)
    Z = func(X, Y)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Podaj nazwe funkcji")
        print("Dostępne funkcje:")
        for func_name in functions:
            print(f" - {func_name}")
        sys.exit(1)

    func_name = sys.argv[1].lower()

    if func_name not in functions:
        print(f"Nieznana funkcja '{func_name}'. Dostępne funkcje to:")
        for func_name in functions:
            print(f" - {func_name}")
        sys.exit(1)

    selected_function = functions[func_name]
    plot_function(selected_function)