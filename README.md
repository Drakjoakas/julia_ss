# julia_ss

En este repositorio se encuentran distintos archivos que explican algunos temas sobre el uso de **Julia**, específicamente orientado al uso de **CUDA.jl** y la programación de GPU's de _nVidia_. 

El repositorio contiene distintos jupyterNotebooks además de algunas implementaciones de algoritmos en paralelo usando **CUDA.jl** o **KernelAbstractions.jl**. 

La estructura del repositorio es la siguiente:

## CUDA_jl.ipynb

Explica las bases y el funcionamiento del paquete (`CUDA.jl`)[https://github.com/JuliaGPU/CUDA.jl], incluyendo la abstracción de más alto nivel, algunos métodos especiales y cómo lanzar kernels, así como una breve introducción al paralelismo.

## LinearAlgebra.ipynb

Contiene la explicación y ejemplos de algunas funciones del paquete de Julia (`LinearAlgebra.jl`)[https://docs.julialang.org/en/v1/stdlib/LinearAlgebra/], el cual permite realizar varias operaciones de álgebra lineal.

## MultipleDispatch.ipynb 

Explicación sencilla del concepto de **despacho múltiple** (multiple dispatch) del lenguaje Julia.

## TiposDeDatos.ipynb

Explicación sencilla del manejo de los **tipos de datos** en el lenguaje Julia.

## Tullio.ipynb

Notebook que contiene una revisión al paquete (`Tullio.jl`)[https://github.com/mcabbott/Tullio.jl], su funcionamiento y significado y algunos ejemplos mostrando la salida para facilitar la comprensión del paquete. Además, se revisa de igual manera funciones y configuración de la biblioteca.

## Cuda jl

Dentro de esta carpeta se encuentran varios archivos con ejemplos de programas escritos en Julia que muestran cómo utilizar CUDA.jl y algunos conceptos de programación en paralelo usando GPU's en Julia.
_Nota: Los ejemplos de Parallel Reduction aun no funcionan. Únicamente la implementación "naive"._

## KernelAbstractions

En la carpeta se encuentran varios archivos que explican cómo utilizar el paquete (`KernelAbstractions.jl`)[https://github.com/JuliaGPU/KernelAbstractions.jl] así como su comparación con CUDA.jl. De igual manera, hay algunos ejemplos sencillos como realizar la transposición de una matriz o multiplicar dos matrices, usando tanto KernelAbstractions como CUDA.jl.

## Parallel Programming

Notebooks del curso (**Parallel Computing**)[https://juliaacademy.com/p/parallel-computing] de **Julia Academy** donde se explican varios conceptos tanto de Julia como de la programaci
