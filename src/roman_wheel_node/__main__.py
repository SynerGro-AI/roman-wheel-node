"""`python -m roman_wheel_node` prints the six original reference calls."""
from .core import roman_wheel_node

if __name__ == "__main__":
    for a, ws in [(1, [0.9, 0.92, 0.88]), (1, [0.7] * 4), (0, [0.95, 0.9]),
                  (1, [0.8, 0.0, 0.9]), (1, [1.5, 0.9]), (1, [])]:
        print(" ", a, ws, roman_wheel_node(a, ws))
