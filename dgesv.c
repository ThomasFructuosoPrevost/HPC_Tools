#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include "mkl_lapacke.h"

// Generate matrix of random values
double *generate_matrix(int size, int seed)
{
    int i;
    double *matrix = (double *)malloc(sizeof(double) * size * size);
    srand(seed);

    for (i = 0; i < size * size; i++)
    {
        matrix[i] = rand() % 100;
    }

    return matrix;
}

// Print a matrix
void print_matrix(const char *name, double *matrix, int size)
{
    int i, j;
    printf("matrix: %s \n", matrix);

    for (i = 0; i < size; i++)
    {
            for (j = 0; j < size; j++)
            {
                printf("%f ", matrix[i * size + j]);
            }
            printf("\n");
    }
}

// Return the absolute of a double
double absolute_value(double n) {
  double res = (n > 0)?n:((-1)*n);
  return res;
}

// Checks if difference between each element of 2 matrices is lower than 0.0005
int check_result(double *bref, double *b, int size) {
    int i;
    for(i=0;i<size*size;i++) {
      if (absolute_value(bref[i] - b[i]) > 0.0005) return 0;
    }
    return 1;
}

// Return the index of the max of n1 and n2
int max(double n1, int indice1, double n2, int indice2) {
    int res = (n1 >= n2)?indice1:indice2;
    return res;
}

// Return the index of the maximum value of a column
int get_max_column(double *A, int size, int j, int r) {
    int indice_max = r+1;
    for (int i = r+2; i < size; i++) {
        indice_max = max(absolute_value(A[indice_max * size + j]), indice_max, absolute_value(A[i * size + j]), i);
    }
    return indice_max;
} 

// Multiply the row k of matrix by val
void multiply_row(double *matrix, int size, int k, double val) {
    for (int j = 0; j < size; j++) {
        matrix[k * size + j] = matrix[k * size + j] * val;
    } 
}

// exchange row indice_row1 and row indice_row2 of matrix
void exchange_row1(double *matrix, int size, int indice_row1, int indice_row2) {
  double tmp;
  for (int j = 0; j < size; j++) {
    tmp = matrix[indice_row1 * size + j];
    matrix[indice_row1 * size + j] = matrix[indice_row2 * size + j];
    matrix[indice_row2 * size + j] = tmp;
  }
}

// Generate identity matrix
void generate_id_matrix(double *matrix, int size) {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {    
            matrix[i * size + j] = (i == j)?1:0;
        }
    }
}

// Substract row indice_row by row indice_r multiplied by val
void Substract_Mult_Val(double *matrix,  int size, int indice_r, int indice_row, double val) {
  for (int j = 0; j < size; j++) {
    matrix[indice_row * size + j] -= matrix[indice_r * size + j] * val;
  }
}

// Apply Gauss-Jordan method
void gauss1(double *matrix, double *I, int size) {
  int r = -1;
  for (int j = 0; j < size; j++) {
    int indice = get_max_column(matrix, size, j, r);
    if (matrix[indice * size + j] != 0) {
      r++;
      float val = 1/matrix[indice * size + j];
      multiply_row(matrix, size, indice, val);
      multiply_row(I, size, indice, val);
      exchange_row1(matrix, size, indice, r);
      exchange_row1(I, size, indice, r);
      for (int i = 0; i < size; i++) {
	if (i != r) {
	  val = matrix[i * size + j];
	  Substract_Mult_Val(matrix, size, r, i, val);
	  Substract_Mult_Val(I, size, r, i, val);
	}
      }
    }
  }
}

// Make matricial product
void matrix_multiplication(double *matrix1, double *matrix2, double *res ,int size) {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            res[i * size + j] = 0;
            for (int k = 0; k < size; k++) {
                res[i * size + j] += matrix1[i * size + k] * matrix2[k * size + j];
            }
        }
    }
}

void my_dgesv(double *a, double *b, double *res, int size) {
  double *Inv_A = (double *)malloc(size * size * sizeof(double));
  generate_id_matrix(Inv_A, size);
  gauss1(a, Inv_A, size);
  matrix_multiplication(Inv_A, b, res, size);
}


    void main(int argc, char *argv[])
    {

        int size = atoi(argv[1]);

        double *a, *aref;
        double *b, *bref;
	

        a = generate_matrix(size, 1);
        aref = generate_matrix(size, 1);        
        b = generate_matrix(size, 2);
        bref = generate_matrix(size, 2);

        //print_matrix("A", a, size);
        //print_matrix("B", b, size);

        // Using MKL to solve the system
        MKL_INT n = size, nrhs = size, lda = size, ldb = size, info;
        MKL_INT *ipiv = (MKL_INT *)malloc(sizeof(MKL_INT)*size);

        clock_t tStart = clock();
        info = LAPACKE_dgesv(LAPACK_ROW_MAJOR, n, nrhs, aref, lda, ipiv, bref, ldb);
        printf("Time taken by MKL: %.2fs\n", (double)(clock() - tStart) / CLOCKS_PER_SEC);
	
        tStart = clock();    
        MKL_INT *ipiv2 = (MKL_INT *)malloc(sizeof(MKL_INT)*size);        
        double *X = (double *)malloc(size * size * sizeof(double));
	my_dgesv(a, b, X, size);
        printf("Time taken by my implementation: %.2fs\n", (double)(clock() - tStart) / CLOCKS_PER_SEC);
        
        if (check_result(bref,X ,size)==1)
            printf("Result is ok!\n");
        else    
            printf("Result is wrong!\n");
        
        //print_matrix("X", X, size);
        //print_matrix("Xref", bref, size);
    }
