import subprocess
import os

# Número de veces que deseas correr el programa
NUM_RUNS = 10

# Nombre de los archivos
source_file = 'reduccionJose.cu'  # Cambia por el nombre de tu archivo CUDA
build_directory = 'tests'           # Carpeta donde se creará el build
output_executable = 'reduccionJoseTests'  # Nombre del ejecutable

# Crear la carpeta de tests si no existe
os.makedirs(build_directory, exist_ok=True)

# Compilar el código CUDA en la carpeta de tests
subprocess.run(
    ['nvcc', '-o', f'{build_directory}/{output_executable}', source_file])

# Archivo de resultados en la carpeta de tests
results_file_path = os.path.join(build_directory, 'testResults.txt')

# Ejecutar el programa varias veces
with open(results_file_path, 'w') as file:
    for run in range(1, NUM_RUNS + 1):
        print(f"Ejecutando prueba {run}...")
        result = subprocess.run([f'./{build_directory}/{output_executable}'],
                                capture_output=True, text=True)
        # Escribe la salida del programa en el archivo
        file.write(result.stdout)
