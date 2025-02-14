#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda.h>
#include <math.h>

// Variables para N y BLOCK_SIZE que se leerán del archivo
int N;          // Número de elementos en el array
int BLOCK_SIZE; // Tamaño del bloque

__global__ void reductionKernel(float *d_input, float *d_output, int n)
{
    extern __shared__ float sharedData[]; // Array compartido en la GPU
    int tid = threadIdx.x;                // ID del hilo
    int index = blockIdx.x * blockDim.x + tid;

    // Cargar datos en el array compartido
    sharedData[tid] = (index < n) ? d_input[index] : 0.0f;
    __syncthreads(); // Sincronizar hilos en el bloque

    // Reducción en el bloque
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2)
    {
        if (tid < stride)
        {
            sharedData[tid] += sharedData[tid + stride];
        }
        __syncthreads(); // Sincronizar hilos en el bloque solo si hay más de un hilo
    }

    // Escribir el resultado del bloque en la salida
    if (tid == 0)
    {
        d_output[blockIdx.x] = sharedData[0];
    }
}

// Función para leer los parámetros desde el archivo de texto
int leerParametros(const char *nombreArchivo)
{
    FILE *archivo = fopen(nombreArchivo, "r");
    if (archivo == NULL)
    {
        printf("No se pudo abrir el archivo %s\n", nombreArchivo);
        return -1;
    }

    // Leer los valores de N y BLOCK_SIZE desde el archivo
    if (fscanf(archivo, "%d", &N) != 1 || fscanf(archivo, "%d", &BLOCK_SIZE) != 1)
    {
        printf("Error al leer los parámetros del archivo.\n");
        fclose(archivo);
        return -1;
    }

    fclose(archivo);
    return 0;
}

int main()
{
    // Leer los parámetros desde el archivo de texto
    if (leerParametros("parameters.txt") != 0)
    {
        return -1;
    }

    // Mostrar el tamaño del array y el número de bloques
    printf("Tamano del array N: %d\n", N);
    printf("Tamano del bloque BLOCK_SIZE: %d\n", BLOCK_SIZE);

    // Inicializar datos
    float *h_input = (float *)malloc(sizeof(float) * N);
    srand(time(NULL)); // Semilla para números aleatorios

    for (int i = 0; i < N; i++)
    {
        h_input[i] = (float)(rand() % 100); // Asignar valores aleatorios entre 0 y 99
    }

    float *d_input, *d_output;
    int numBlocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;            // Número de bloques
    float *h_output = (float *)malloc(sizeof(float) * numBlocks); // Buffer de salida

    // Asignar memoria en la GPU
    cudaMalloc(&d_input, sizeof(float) * N);
    cudaMalloc(&d_output, sizeof(float) * numBlocks);

    // Copiar datos a la GPU
    cudaMemcpy(d_input, h_input, sizeof(float) * N, cudaMemcpyHostToDevice);

    // Crear eventos para temporización
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Iniciar temporización
    cudaEventRecord(start, 0);

    // Configurar el tamaño de los bloques y la grilla
    dim3 blockSize(BLOCK_SIZE);
    dim3 gridSize(numBlocks);

    // Lanzar el kernel con memoria compartida dinámica
    reductionKernel<<<gridSize, blockSize, BLOCK_SIZE * sizeof(float)>>>(d_input, d_output, N);
    cudaDeviceSynchronize(); // Esperar a que el kernel termine de ejecutarse

    // Detener temporización
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    // Calcular el tiempo transcurrido
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Copiar el resultado de vuelta a la CPU
    cudaMemcpy(h_output, d_output, sizeof(float) * numBlocks, cudaMemcpyDeviceToHost);

    // Sumar los resultados de los bloques
    float totalSum = 0.0f;
    for (int i = 0; i < numBlocks; i++)
    {
        totalSum += h_output[i];
    }

    // Calcular la suma secuencial para validar la implementación
    float sequentialSum = 0.0f;
    for (int i = 0; i < N; i++)
    {
        sequentialSum += h_input[i];
    }

    // Mostrar los resultados
    printf("Suma total (CUDA): %f\n", totalSum);
    printf("Suma total (secuencial): %f\n", sequentialSum);
    printf("Tiempo de ejecucion (CUDA): %f ms\n", milliseconds);

    // Validar el resultado
    if (fabs(totalSum - sequentialSum) < 1e-5) // Comparar con tolerancia
    {
        printf("La implementacion es valida.\n");
    }
    else
    {
        printf("La implementacion es invalida.\n");
    }

    // Liberar memoria
    cudaFree(d_input);
    cudaFree(d_output);
    free(h_input);
    free(h_output);

    return 0;
}
