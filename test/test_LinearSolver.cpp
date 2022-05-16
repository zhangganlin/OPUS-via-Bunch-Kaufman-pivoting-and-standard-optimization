#include "LinearSolver.h"
#include <iostream>
#include <cstring>
using namespace std;



void test_solve_diag(){
    /*
    A =
        3     0     0     0     0
        0     4     3     0     0
        0     3     1     0     0
        0     0     0     5     0
        0     0     0     0     6

    b = 
        [3     4     5     6     7].T
        
    x should be: [1.0000    2.2000   -1.6000    1.2000    1.1667].T
    */
    int n = 5;
    double* D = (double*)malloc(25 * sizeof(double));
    double* b = (double*)malloc(5 * sizeof(double));
    double* x = (double*)malloc(5 * sizeof(double));
    int* pivot = (int*)malloc(5*sizeof(int));
    for(int i = 0; i < 25; i++){
        D[i] = 0;        
    }
    D[0] = 3;
    D[1 * 5 + 1] = 4; D[1 * 5 + 2] = 3;
    D[2 * 5 + 1] = 3; D[2 * 5 + 2] = 1;
    D[3 * 5 + 3] = 5; D[4 * 5 + 4] = 6;
    for(int i = 3; i < 8; i++){
        b[i-3] = i;
    }
    pivot[0] = 1;pivot[1] = 2;pivot[2] = 0;pivot[3] = 1;pivot[4] = 1;
    solve_diag(D,pivot, x, b, n);
    cout << "x should be:\n";
    cout << "1 2.2 -1.6 1.2 1.16667\n";
    cout << "solved x:\n";
    for(int i = 0; i < n; i++){
        cout << x[i] << " ";
    }
    cout << endl;
    free(D);
    free(b);
    free(x);
    free(pivot);
}   

void test_solve_lower(){
    /*
    A =
     3     0     0     0     0
     7     4     0     0     0
     8     3     1     0     0
     9     8     7     5     0
     9     8     7     5    19
    b = [3,4,5,6,7].T
    x should be: [1.0000, -0.7500, -0.7500, 1.6500, 0.0526].T
    */
    int n = 5;
    double* D = (double*)malloc(25 * sizeof(double));
    double* b = (double*)malloc(5 * sizeof(double));
    double* x = (double*)malloc(5 * sizeof(double));
    for(int i = 0; i < 25; i++){
        D[i] = 0;        
    }
    D[0] = 3;
    D[1 * 5 + 0] = 7; D[1 * 5 + 1] = 4;
    D[2 * 5 + 0] = 8; D[2 * 5 + 1] = 3; D[2 * 5 + 2] = 1; 
    D[3 * 5 + 0] = 9; D[3 * 5 + 1] = 8; D[3 * 5 + 2] = 7; D[3 * 5 + 3] = 5;
    D[4 * 5 + 0] = 9; D[4 * 5 + 1] = 8; D[4 * 5 + 2] = 7; D[4 * 5 + 3] = 5; D[4 * 5 + 4] = 19;
    for(int i = 0; i < 5; i++){
        b[i] = i+3;
    }
    solve_lower(D,x,b,n);
    cout << "x should be:\n";
    cout << "1.0000, -0.7500, -0.7500, 1.6500, 0.0526\n";
    cout << "solved x:\n";
    for(int i = 0; i < n; i++){
        cout << x[i] << " ";
    }
    cout << endl;
    free(D);
    free(x);
    free(b);
}

void test_solve_upper(){
    /*
    A =
     3     0     0     0     0
     7     4     0     0     0
     8     3     1     0     0
     9     8     7     5     0
     9     8     7     5    19
    b = [3,4,5,6,7].T
    x should be: [3.7833, 1.1500, -3.4000, 0.8316, 0.3684].T
    */
    int n = 5;
    double* D = (double*)malloc(25 * sizeof(double));
    double* b = (double*)malloc(5 * sizeof(double));
    double* x = (double*)malloc(5 * sizeof(double));
    for(int i = 0; i < 25; i++){
        D[i] = 0;        
    }
    D[0] = 3;
    D[1 * 5 + 0] = 7; D[1 * 5 + 1] = 4;
    D[2 * 5 + 0] = 8; D[2 * 5 + 1] = 3; D[2 * 5 + 2] = 1; 
    D[3 * 5 + 0] = 9; D[3 * 5 + 1] = 8; D[3 * 5 + 2] = 7; D[3 * 5 + 3] = 5;
    D[4 * 5 + 0] = 9; D[4 * 5 + 1] = 8; D[4 * 5 + 2] = 7; D[4 * 5 + 3] = 5; D[4 * 5 + 4] = 19;
    for(int i = 0; i < 5; i++){
        b[i] = i+3;
    }
    solve_upper(D,x,b,n);
    cout << "x should be:\n";
    cout << "3.7833, 1.1500, -3.4000, 0.8316, 0.3684\n";
    cout << "solved x:\n";
    for(int i = 0; i < n; i++){
        cout << x[i] << " ";
    }
    cout << endl;
    free(D);
    free(x);
    free(b);
}

void test_solve_BunchKaufman(){

    /*
    D = 
     3     0     0     0     0
     0     4     9     0     0
     0     9     5     0     0
     0     0     0     6     0
     0     0     0     0     7

    L =
     3     0     0     0     0
     7     4     0     0     0
     8     3     1     0     0
     9     8     7     5     0
     9     8     7     5    19

    P = 
     0     1     0     0     0
     1     0     0     0     0
     0     0     1     0     0
     0     0     0     0     1
     0     0     0     1     0

    b = [3,4,5,6,7].T
    x should be [0.4339, 1.0693, -0.8307, -0.0004, 0.0943].T

    */

    int* P = (int*)malloc(5*sizeof(int));
    int* pivot = (int*)malloc(5*sizeof(int));
    double* L = (double*)malloc(25 * sizeof(double));
    double* b = (double*)malloc(5 * sizeof(double));
    double* x = (double*)malloc(5 * sizeof(double));
    double* D = (double*)malloc(25 * sizeof(double));
    for(int i = 0; i < 25; i++){
        L[i] = 0;        
    }
    L[0] = 3;
    L[1 * 5 + 0] = 7; L[1 * 5 + 1] = 4;
    L[2 * 5 + 0] = 8; L[2 * 5 + 1] = 3; L[2 * 5 + 2] = 1; 
    L[3 * 5 + 0] = 9; L[3 * 5 + 1] = 8; L[3 * 5 + 2] = 7; L[3 * 5 + 3] = 5;
    L[4 * 5 + 0] = 9; L[4 * 5 + 1] = 8; L[4 * 5 + 2] = 7; L[4 * 5 + 3] = 5; L[4 * 5 + 4] = 19;
    for(int i = 0; i < 5; i++){
        b[i] = i+3;
        D[i*5+i] = i + 3;
    }
    D[5*2 + 1] = D[5*1 + 2] = 9;

    P[0] = 2; P[1]=1; P[2]=3; P[3]=5; P[4] = 4;
    pivot[0] = 1;pivot[1] = 2;pivot[2] = 0;pivot[3] = 1;pivot[4] = 1;

    int n = 5;
    double* Pb = (double*)malloc(n*sizeof(double));
    double* DLTPx = (double*)malloc(n*sizeof(double));
    double* LTPx = (double*)malloc(n*sizeof(double));
    double* Px = (double*)malloc(n*sizeof(double));
    for(int i = 0; i < n; i++){
        Pb[i] = b[P[i]-1];
    }

    solve_lower(L,DLTPx,Pb,n);
    solve_diag(D,pivot,LTPx,DLTPx,n); //Assume D is saved in A
    solve_upper(L,Px,LTPx,n);
    for(int i = 0; i < n; i++){
        x[P[i]-1] = Px[i];
    }
    cout << "x should be:\n0.433896 1.06931 -0.830719 -0.000395726 0.0942846\n";
    cout << "solved x:\n";
    for(int i = 0; i < n; i++){
        cout << x[i] << " ";
    }
    cout << endl;


    free(D);
    free(P);
    free(L);
    free(x);
    free(b);
    free(pivot);
    free(Pb);free(DLTPx);free(Px);free(LTPx);
    
}

void test_LU_solver(){
    /*
    A =
         211          63         252         569         569
          63          27          72          81          81
         252          72         287         608         608
         569          81         608        4429        1902
         569          81         608        1902        1902
    b = [3,4,5,6,7].T
    x should be: [0.4339, 1.0693, -0.8307, -0.0004, 0.0943].T
    */
    double* A = (double*)malloc(25 * sizeof(double));  
    double* b = (double*)malloc(5 * sizeof(double)); 
    double* x = (double*)malloc(5 * sizeof(double));
    A[0 * 5 + 0] = 211; A[0 * 5 + 1] = 63; A[0 * 5 + 2] = 252; A[0 * 5 + 3] = 569 ; A[0 * 5 + 4] = 569;
    A[1 * 5 + 0] = 63 ; A[1 * 5 + 1] = 27; A[1 * 5 + 2] = 72 ; A[1 * 5 + 3] = 81  ; A[1 * 5 + 4] = 81;
    A[2 * 5 + 0] = 252; A[2 * 5 + 1] = 72; A[2 * 5 + 2] = 287; A[2 * 5 + 3] = 608 ; A[2 * 5 + 4] = 608;
    A[3 * 5 + 0] = 569; A[3 * 5 + 1] = 81; A[3 * 5 + 2] = 608; A[3 * 5 + 3] = 4429; A[4 * 5 + 4] = 1902;
    A[4 * 5 + 0] = 569; A[4 * 5 + 1] = 81; A[4 * 5 + 2] = 608; A[4 * 5 + 3] = 1902; A[5 * 5 + 4] = 1902;
    
    for(int i = 0; i < 5; i++){
        b[i] = i+3;
    }

    LUdecomp(A,b,x,3,2);
    cout << "solved x:\n";
    for(int i = 0; i < 5; i++){
        cout << x[i] << " ";
    }
    cout << endl;
    free(A);
    free(x);
    free(b);
}


void testBunchKaufman1(){
    const int M = 3;
    double* A = (double*)malloc((M * M) * sizeof(double));
    double* L = (double*)malloc((M * M) * sizeof(double));
    int* P = (int*)malloc(M * sizeof(int));
    int* pivot = (int*)malloc(M * sizeof(int));

    A[0] = 27;    A[1] = 98;    A[2] = 49;
    A[3] = 98;    A[4] = 83;    A[5] = 54;
    A[6] = 49;    A[7] = 54;    A[8] = 37;

    BunchKaufman(A, L, P, pivot, M);
    printf("L \n");
    for(int i = 0; i < M; i++) {
        for(int j = 0; j < M; j++) {
            printf("%f ", L[i * M + j]);
        }
        printf("\n");
    } 
    printf("D \n");
    for(int i = 0; i < M; i++) {
        for(int j = 0; j < M; j++) {
            printf("%f ", A[i * M + j]);
        }
        printf("\n");
    }
    printf("P \n");
    for(int j = 0; j < M; j++) {
        printf("%d ", P[j]);
    }
    printf("\npiviot\n");
    for(int j = 0; j < M; j++) {
        printf("%d ", pivot[j]);
    }
    printf("\n");

    free(A);
    free(L);
    free(P);
    free(pivot);
}
void testBunchKaufman2(){
    const int M = 5;
    double* A = (double*)malloc((M * M) * sizeof(double));
    double* L = (double*)malloc((M * M) * sizeof(double));
    int* P = (int*)malloc(M * sizeof(int));
    int* pivot = (int*)malloc(M * sizeof(int));
    A[0]  = 211; A[1]  = 63; A[2]  = 252; A[3]  =  569; A[4]  =  569;
    A[5]  =  63; A[6]  = 27; A[7]  =  72; A[8]  =   81; A[9]  =   81;
    A[10] = 252; A[11] = 72; A[12] = 287; A[13] =  608; A[14] =  608;
    A[15] = 569; A[16] = 81; A[17] = 608; A[18] = 4429; A[19] = 1902;
    A[20] = 569; A[21] = 81; A[22] = 608; A[23] = 1902; A[24] = 1902;
    BunchKaufman(A, L, P, pivot, M);
    printf("L \n");
    for(int i = 0; i < M; i++) {
        for(int j = 0; j < M; j++) {
            printf("%f ", L[i * M + j]);
        }
        printf("\n");
    } 
    printf("D \n");
    for(int i = 0; i < M; i++) {
        for(int j = 0; j < M; j++) {
            printf("%f ", A[i * M + j]);
        }
        printf("\n");
    }
    printf("P \n");
    for(int j = 0; j < M; j++) {
        printf("%d ", P[j]);
    }
    printf("\npiviot\n");
    for(int j = 0; j < M; j++) {
        printf("%d ", pivot[j]);
    }
    printf("\n");
    free(A);
    free(L);
    free(P);
    free(pivot);
}

void testBunchKaufman3(){
    int N = 2;
    int func_dim = 1;
    int d = 1;
    const int M = N + d + 1;
    double* A = (double*)malloc((M * M) * sizeof(double));
    double* L = (double*)malloc((M * M) * sizeof(double));
    int* P = (int*)malloc(M * sizeof(int));
    int* pivot = (int*)malloc(M * sizeof(int));
    int i, j;
    A[0]  =  6;  A[1]  =  12;  A[2]  =   3;  A[3]  = -6;
    A[4]  = 12;  A[5]  =  -8;  A[6]  = -13;  A[7]  =  4;
    A[8]  =  3;  A[9]  = -13;  A[10] =  -7;  A[11] =  1;
    A[12] = -6;  A[13] =   4;  A[14] =   1;  A[15] =  6;
    BunchKaufman(A, L, P, pivot, M);
    printf("L \n");
    for(int i = 0; i < M; i++) {
        for(int j = 0; j < M; j++) {
            printf("%f ", L[i * M + j]);
        }
        printf("\n");
    } 
    printf("D \n");
    for(int i = 0; i < M; i++) {
        for(int j = 0; j < M; j++) {
            printf("%f ", A[i * M + j]);
        }
        printf("\n");
    }
    printf("P \n");
    for(int j = 0; j < M; j++) {
        printf("%d ", P[j]);
    }
    printf("\npiviot\n");
    for(int j = 0; j < M; j++) {
        printf("%d ", pivot[j]);
    }
    printf("\n");

    free(A);
    free(L);
    free(P);
    free(pivot);
}

void testBunchKaufman4(){
    const int M = 5;
    double* A = (double*)malloc((M * M) * sizeof(double));
    double* L = (double*)malloc((M * M) * sizeof(double));
    int* P = (int*)malloc(M * sizeof(int));
    int* pivot = (int*)malloc(M * sizeof(int));
    A[0]  = 211; A[1]  = 63; A[2]  = 252; A[3]  =  569; A[4]  =  569;
    A[5]  =  63; A[6]  = 27; A[7]  =   0; A[8]  =   81; A[9]  =   81;
    A[10] = 252; A[11] =  0; A[12] = 287; A[13] =  608; A[14] =  608;
    A[15] = 569; A[16] = 81; A[17] = 608; A[18] =    0; A[19] = 1902;
    A[20] = 569; A[21] = 81; A[22] = 608; A[23] = 1902; A[24] = 1902;
    BunchKaufman(A, L, P, pivot, M);
    printf("L \n");
    for(int i = 0; i < M; i++) {
        for(int j = 0; j < M; j++) {
            printf("%f ", L[i * M + j]);
        }
        printf("\n");
    } 
    printf("D \n");
    for(int i = 0; i < M; i++) {
        for(int j = 0; j < M; j++) {
            printf("%f ", A[i * M + j]);
        }
        printf("\n");
    }
    printf("P \n");
    for(int j = 0; j < M; j++) {
        printf("%d ", P[j]);
    }
    printf("\npiviot\n");
    for(int j = 0; j < M; j++) {
        printf("%d ", pivot[j]);
    }
    printf("\n");
    free(A);
    free(L);
    free(P);
    free(pivot);
}

void test_BunchKaufmanAndSolver1(){
    const int M = 5;
    double* A = (double*)malloc((M * M) * sizeof(double));
    double* x = (double*)malloc(M * sizeof(double));
    double* b = (double*)malloc(M * sizeof(double));

    A[0]  = 211; A[1]  = 63; A[2]  = 252; A[3]  =  569; A[4]  =  569;
    A[5]  =  63; A[6]  = 27; A[7]  =   0; A[8]  =   81; A[9]  =   81;
    A[10] = 252; A[11] =  0; A[12] = 287; A[13] =  608; A[14] =  608;
    A[15] = 569; A[16] = 81; A[17] = 608; A[18] =    0; A[19] = 1902;
    A[20] = 569; A[21] = 81; A[22] = 608; A[23] = 1902; A[24] = 1902;
    for(int i = 0; i < M; i++){
        b[i] = i+3;
    }

    solve_BunchKaufman(A,x,b,5);
    cout << "x should be:\n0.0778 -0.0039 -0.0301 0.0005 -0.0103\n";
    cout << "solved x:\n";
    for(int i = 0; i < M; i++){
        cout << x[i] << " ";
    }
    cout << endl;

    free(A);
    free(x);
    free(b);
}


void test_BunchKaufmanAndSolver2(){
    const int M = 15;
    double* A = (double*)malloc((M * M) * sizeof(double));
    double* x = (double*)malloc(M * sizeof(double));
    double* b = (double*)malloc(M * sizeof(double));

    A[0] = 0.000000; A[1] = 0.776324; A[2] = 20.840404; A[3] = 0.348433; A[4] = 1.500600; A[5] = 0.659031; A[6] = 0.157198; A[7] = 1.257581; A[8] = 11.018425; A[9] = 3.895601; A[10] = 9.041188; A[11] = 8.769512; A[12] = -0.853333; A[13] = 0.682667; A[14] = 1.000000; 
    A[15] = 0.776324; A[16] = 0.000000; A[17] = 21.444451; A[18] = 0.014060; A[19] = 8.769512; A[20] = 3.895601; A[21] = 0.233003; A[22] = 3.100491; A[23] = 18.598233; A[24] = 15.344400; A[25] = 26.608982; A[26] = 21.931813; A[27] = -1.194667; A[28] = 1.536000; A[29] = 1.000000; 
    A[30] = 20.840404; A[31] = 21.444451; A[32] = 0.000000; A[33] = 18.598233; A[34] = 28.056598; A[35] = 10.249877; A[36] = 13.432248; A[37] = 4.971027; A[38] = 1.757523; A[39] = 44.789680; A[40] = 35.810416; A[41] = 11.805275; A[42] = 1.536000; A[43] = 2.048000; A[44] = 1.000000; 
    A[45] = 0.348433; A[46] = 0.014060; A[47] = 18.598233; A[48] = 0.000000; A[49] = 6.210593; A[50] = 2.368323; A[51] = 0.055578; A[52] = 1.918050; A[53] = 14.590726; A[54] = 11.805275; A[55] = 20.840404; A[56] = 16.777216; A[57] = -1.024000; A[58] = 1.365333; A[59] = 1.000000; 
    A[60] = 1.500600; A[61] = 8.769512; A[62] = 28.056598; A[63] = 6.210593; A[64] = 0.000000; A[65] = 0.776324; A[66] = 3.895601; A[67] = 3.100491; A[68] = 9.041188; A[69] = 0.157198; A[70] = 0.899852; A[71] = 2.605051; A[72] = -0.341333; A[73] = -0.341333; A[74] = 1.000000; 
    A[75] = 0.659031; A[76] = 3.895601; A[77] = 10.249877; A[78] = 2.368323; A[79] = 0.776324; A[80] = 0.000000; A[81] = 0.899852; A[82] = 0.157198; A[83] = 2.787465; A[84] = 3.100491; A[85] = 3.895601; A[86] = 1.864022; A[87] = 0.000000; A[88] = 0.512000; A[89] = 1.000000; 
    A[90] = 0.157198; A[91] = 0.233003; A[92] = 13.432248; A[93] = 0.055578; A[94] = 3.895601; A[95] = 0.899852; A[96] = 0.000000; A[97] = 0.659031; A[98] = 8.950323; A[99] = 8.679568; A[100] = 14.590726; A[101] = 10.440274; A[102] = -0.682667; A[103] = 1.194667; A[104] = 1.000000; 
    A[105] = 1.257581; A[106] = 3.100491; A[107] = 4.971027; A[108] = 1.918050; A[109] = 3.100491; A[110] = 0.157198; A[111] = 0.659031; A[112] = 0.000000; A[113] = 1.757523; A[114] = 7.971260; A[115] = 8.679568; A[116] = 3.164417; A[117] = 0.170667; A[118] = 1.024000; A[119] = 1.000000; 
    A[120] = 11.018425; A[121] = 18.598233; A[122] = 1.757523; A[123] = 14.590726; A[124] = 9.041188; A[125] = 2.787465; A[126] = 8.950323; A[127] = 1.757523; A[128] = 0.000000; A[129] = 16.331818; A[130] = 9.779026; A[131] = 1.257581; A[132] = 1.365333; A[133] = 0.853333; A[134] = 1.000000; 
    A[135] = 3.895601; A[136] = 15.344400; A[137] = 44.789680; A[138] = 11.805275; A[139] = 0.157198; A[140] = 3.100491; A[141] = 8.679568; A[142] = 7.971260; A[143] = 16.331818; A[144] = 0.000000; A[145] = 0.659031; A[146] = 4.749017; A[147] = -0.512000; A[148] = -0.853333; A[149] = 1.000000; 
    A[150] = 9.041188; A[151] = 26.608982; A[152] = 35.810416; A[153] = 20.840404; A[154] = 0.899852; A[155] = 3.895601; A[156] = 14.590726; A[157] = 8.679568; A[158] = 9.779026; A[159] = 0.659031; A[160] = 0.000000; A[161] = 1.305034; A[162] = 0.341333; A[163] = -1.024000; A[164] = 1.000000; 
    A[165] = 8.769512; A[166] = 21.931813; A[167] = 11.805275; A[168] = 16.777216; A[169] = 2.605051; A[170] = 1.864022; A[171] = 10.440274; A[172] = 3.164417; A[173] = 1.257581; A[174] = 4.749017; A[175] = 1.305034; A[176] = 0.000000; A[177] = 1.024000; A[178] = -0.170667; A[179] = 1.000000; 
    A[180] = -0.853333; A[181] = -1.194667; A[182] = 1.536000; A[183] = -1.024000; A[184] = -0.341333; A[185] = 0.000000; A[186] = -0.682667; A[187] = 0.170667; A[188] = 1.365333; A[189] = -0.512000; A[190] = 0.341333; A[191] = 1.024000; A[192] = 0.000000; A[193] = 0.000000; A[194] = 0.000000; 
    A[195] = 0.682667; A[196] = 1.536000; A[197] = 2.048000; A[198] = 1.365333; A[199] = -0.341333; A[200] = 0.512000; A[201] = 1.194667; A[202] = 1.024000; A[203] = 0.853333; A[204] = -0.853333; A[205] = -1.024000; A[206] = -0.170667; A[207] = 0.000000; A[208] = 0.000000; A[209] = 0.000000; 
    A[210] = 1.000000; A[211] = 1.000000; A[212] = 1.000000; A[213] = 1.000000; A[214] = 1.000000; A[215] = 1.000000; A[216] = 1.000000; A[217] = 1.000000; A[218] = 1.000000; A[219] = 1.000000; A[220] = 1.000000; A[221] = 1.000000; A[222] = 0.000000; A[223] = 0.000000; A[224] = 0.000000; 
    b[0]=0.000000; b[1]=0.000000; b[2]=0.000000; b[3]=0.000000; b[4]=0.000000; b[5]=0.000000; b[6]=0.000000; b[7]=0.000000; b[8]=0.000000; b[9]=0.000000; b[10]=0.000000; b[11]=0.000000; b[12]=0.000000; b[13]=0.000000; b[14]=0.000000; 

    solve_BunchKaufman(A,x,b,M);
    cout << "x should be:\n0 -0 -0 -0 -0 0 -0 0 -0 -0 -0 -0 -0 0 0 \n";
    cout << "solved x:\n";
    for(int i = 0; i < M; i++){
        cout << x[i] << " ";
    }
    cout << endl;

    free(A);
    free(x);
    free(b);
}


int main(){
    // test_solve_diag();
    // test_solve_lower();
    // test_solve_upper();
    // test_solve_BunchKaufman();
    // test_LU_solver();
    // testBunchKaufman1();
    // testBunchKaufman2();
    // testBunchKaufman3();
    // testBunchKaufman4();
    test_BunchKaufmanAndSolver1();
}