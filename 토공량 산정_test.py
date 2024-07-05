from typing import List
from scipy.interpolate import griddata
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

def validate_input(file_path: str, file_name: str) -> bool:
    if not file_path or not file_name:
        raise ValueError("Error: Both file path and file name are required.")
    if not os.path.exists(os.path.join(file_path, file_name)):
        raise FileNotFoundError(f"Error: File '{file_name}' not found in the specified path.")
    return True

def calculate_soil_volume(x: List[float], y: List[float], z: List[float], target_z: float) -> tuple:
    if len(x) != len(y) or len(y) != len(z):
        raise ValueError("Error: X, Y, and Z lists must have the same length.")
    
    upper_volume = 0
    lower_volume = 0
    
    for i in range(len(x)-1):
        dx = abs(x[i] - x[i+1])
        dy = abs(y[i] - y[i+1])
        
        if z[i] >= target_z and z[i+1] >= target_z:
            upper_volume += dx * dy * (z[i+1] - z[i])
        elif z[i] < target_z and z[i+1] < target_z:
            lower_volume += dx * dy * (z[i+1] - z[i])
        else:
            upper_dz = max(0, target_z - min(z[i], z[i+1]))
            lower_dz = max(0, max(z[i], z[i+1]) - target_z)
            upper_volume += dx * dy * upper_dz
            lower_volume += dx * dy * lower_dz
    
    total_volume = upper_volume - lower_volume
    return upper_volume, lower_volume, total_volume

def find_optimal_target_z(x: List[float], y: List[float], z: List[float]) -> tuple:
    min_z = min(z)
    max_z = max(z)
    
    optimal_target_z = 0
    total_optimal_volume = 0
    for target_z in np.linspace(min_z, max_z, 100):
        result = calculate_soil_volume(x, y, z, target_z)
        if result is not None:
            upper_volume, lower_volume, total_volume = result
            if total_volume > total_optimal_volume:
                optimal_target_z = target_z
                total_optimal_volume = total_volume
    
    return optimal_target_z, total_optimal_volume

def plot_soil_volume(x: List[float], y: List[float], z: List[float], volume: float, target_z: float) -> None:
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122)
    
    ax.scatter(x, y, z, c='b', marker='o')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Scatter Plot')
    
    xx, yy = np.meshgrid(np.linspace(min(x), max(x), 50), np.linspace(min(y), max(y), 50))
    zz = np.full_like(xx, target_z)
    ax.plot_surface(xx, yy, zz, alpha=0.2, color='g')
    
    zz = griddata((x, y), z, (xx, yy), method='cubic')
    
    contour = ax2.contourf(xx, yy, zz, levels=[min(z), target_z, max(z)], colors=['blue', 'green'])
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_title('Contour Plot')
    
    # Add a colorbar to the contour plot
    fig.colorbar(contour, ax=ax2, orientation='vertical')
    
    plt.tight_layout()
    plt.show()

def read_excel_data(file_path: str, file_name: str) -> tuple:
    file_full_path = os.path.join(file_path, file_name)
    df = pd.read_excel(file_full_path)
    
    x = df['X'].tolist()
    y = df['Y'].tolist()
    z = df['Z'].tolist()
    
    return x, y, z

file_path = input("Enter the file path: ")
file_name = input("Enter the file name: ")

try:
    validate_input(file_path, file_name)
except (ValueError, FileNotFoundError) as e:
    print(e)
    exit()

x, y, z = read_excel_data(file_path, file_name)

optimal_target_z, total_optimal_volume = find_optimal_target_z(x, y, z)

print(f"최적 목표 높이 Z: {optimal_target_z}")
print(f"총 최적 체적: {total_optimal_volume}")

upper_volume, lower_volume, _ = calculate_soil_volume(x, y, z, optimal_target_z)

print(f"최적 목표 높이 이상의 토사의 체적: {upper_volume}")
print(f"최적 목표 높이 미만의 토사의 체적: {lower_volume}")

plot_soil_volume(x, y, z, total_optimal_volume, optimal_target_z)

