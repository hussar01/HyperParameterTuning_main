import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
print("Current Matplotlib backend:", matplotlib.get_backend())

def plot_sine_wave():
    # Generate x values from 0 to 2π
    x = np.linspace(0, 2 * np.pi, 100)
    # Calculate the sine of each x value
    y = np.sin(x)

    # Create the plot
    plt.plot(x, y, label="Sine Wave")
    plt.xlabel("x")
    plt.ylabel("sin(x)")
    plt.title("Plot of the Sine Wave")
    plt.legend()

    # Display the plot
    plt.show()
'''
1=98.96,
2=98.86,
3=98.79,
4=98.39,
5=98.85,
6=99.10
7=99.00,
8=99.04,
9=98.74,
10=98.80
'''
def calculation():


    # Replace this list with your own samples
    samples = [98.39, 98.74, 98.79, 98.80, 98.85, 98.86, 98.96, 99.00, 99.04, 99.10]
    # Calculate max, min, mean, and standard deviation
    max_value = np.max(samples)
    min_value = np.min(samples)
    mean_value = np.mean(samples)
    std_dev = np.std(samples)

    # Display calculated statistics
    print("\n--- Statistics ---")
    print(f"Max: {max_value}")
    print(f"Min: {min_value}")
    print(f"Mean: {mean_value}")
    print(f"Standard Deviation: {std_dev}")

    # Plot the samples
    plt.figure(figsize=(10, 6))
    plt.plot(samples, marker='o', label='Samples')

    # Add horizontal lines for mean, max, min, and standard deviation
    plt.axhline(mean_value, color='r', linestyle='--', label=f"Mean ({mean_value:.2f})")
    plt.axhline(max_value, color='g', linestyle='--', label=f"Max ({max_value:.2f})")
    plt.axhline(min_value, color='b', linestyle='--', label=f"Min ({min_value:.2f})")
    plt.axhline(mean_value + std_dev, color='orange', linestyle=':',
                label=f"Mean + Std Dev ({mean_value + std_dev:.2f})")
    plt.axhline(mean_value - std_dev, color='purple', linestyle=':',
                label=f"Mean - Std Dev ({mean_value - std_dev:.2f})")

    # Add title, labels, and legend
    plt.title("Sample Analysis")
    plt.xlabel("Sample Index")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)

    # Show the plot
    plt.show()


if __name__ == "__main__":
    #plot_sine_wave()
    calculation()
