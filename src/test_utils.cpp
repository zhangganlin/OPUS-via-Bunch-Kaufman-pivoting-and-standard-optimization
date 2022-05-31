#include "test_utils.h"
#include "tsc_x86.h"
using namespace std;

void read_matrix(double* A, int m, int n, string path){
    std::ifstream in(path);
    std::string line;
    
    int i = 0, k = 0;

    while (std::getline(in, line))
    {
        double value;
        int k = 0;
        std::stringstream ss(line);

        while (ss >> value)
        {
            A[i*n+k] = value;
            ++k;
        }
        ++i;
    }
}

void generate_random_dense(double* arr, int m, int n){
    for(int i = 0; i < m; i++){
		for(int j =0; j < n; j++)
        	arr[i*n+j] = (double)rand()/ (double)RAND_MAX;
    }
}

void generate_random_l(double* arr, int n){
    for(int i = 0; i < n; i++){
		arr[i*n+i] = 1;
		for(int j = 0; j < i; j++)
        	arr[i*n+j] = (double)rand()/ (double)RAND_MAX;
    }
}

void generate_random_symmetry(double* arr, int n){
    for(int i = 0; i < n; i++){
		arr[i*n+i] = (double)rand()/ (double)RAND_MAX;
		for(int j = 0; j < i; j++){
        	arr[i*n+j] = (double)rand()/ (double)RAND_MAX;
        	arr[j*n+i] = arr[i*n+j];
        }
    }
}


void generate_random_d(double* arr, int n, int* pivot){
    for(int i = 0; i < n; i++){
		if (pivot[i]==1){
			arr[i*n+i] = (double)rand()/ (double)RAND_MAX;
		}else if(pivot[i]==2){
			arr[i*n+i] = (double)rand()/ (double)RAND_MAX;
			arr[(i+1)*n + i+1] = (double)rand()/ (double)RAND_MAX;

			arr[i*n + i+1] = (double)rand()/ (double)RAND_MAX;
			arr[(i+1)*n + i] = arr[i*n + i+1];
		}
    }
}

// A: m*n, B:n*p
void matrix_mul(double*A, double*B, double* res, int m, int n, int p){
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < p; j++) {
			for (int a = 1; a <= n; a++) {
				res[i*p+j] +=  A[i*n +a - 1] * B[(a - 1)*p  + j ];
			}
		}
	}
}


void matrix_transpose(double*A, double*At, int n){
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			At[i*n+j] = A[j*n+i];
		}
	}
}

void matrix_transpose(double*A, double*At, int m, int n){
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < n; j++) {
			At[i*n+j] = A[j*m+i];
		}
	}
}

//A[start_row:start_row+n_row, start_col:start_col+n_col] = x
void matrix_set_block(double*A, int start_row, int start_col, int n_row, int n_col, double*x, int n){
	for (int i = 0; i < n_row; i++) {
		for (int j = 0; j < n_col; j++) {
			A[(i+start_row)*n + j+start_col] = x[i*n_row+j];
		}
	}
}

//x = A[start_row:start_row+n_row, start_col:start_col+n_col]
void matrix_get_block(double*A, int start_row, int start_col, int n_row, int n_col, double*x, int n){
	for (int i = 0; i < n_row; i++) {
		for (int j = 0; j < n_col; j++) {
			x[i*n_row+j] = A[(i+start_row)*n + j+start_col];
		}
	}
}


void print_matrix(double* A, int m, int n, int precision){
	for(int i = 0; i < m; i++){
		for(int j = 0; j < n; j++){
			cout << fixed<< setprecision(precision)<< A[i*n+j] << " ";
		}
		cout << endl;
	}
}

void print_vector(int* b, int n){
    for(int i = 0; i < n; i++){
        cout << b[i] << " ";
    }
    cout << endl;
}

bool compare_matrix(double* A1, double* A2, int m, int n){
	double threshold = 1e-5;
	for(int i = 0; i < m; i++){
		for(int j = 0; j < n; j++){
			if (abs(A1[i*n+j]-A2[i*n+j])>threshold){
				return false;
			}
		}
	}
	return true;
}


unsigned long long& flops() {
	static myInt64 flop;
	return flop;
}

void cycle_stastic_init(stastic_t& obj){
    obj.step1to4 = 0;
    obj.step5_time.clear();
    obj.step5_x_history_size.clear();
    obj.step6a.clear();
    obj.step6b.clear();
    obj.step7.clear();
    obj.step8.clear();
    obj.step9_time.clear();
    obj.step9_x_history_size.clear();
    obj.step10.clear();
    obj.step11.clear();
}

void print_stastic(stastic_t& obj, opus_settings_t *settings){
    std::cout << "parameters:" << std::endl;
    std::cout << "  dim: " << settings->dim << std::endl;
    std::cout << "  range_lo: " << settings->range_lo[0] << std::endl;
    std::cout << "  range_hi: " << settings->range_hi[0] << std::endl;
    std::cout << "  swarm_size: " << settings->size << std::endl;
    std::cout << "  k_size: " << settings->k_size << std::endl;
    std::cout << "  opt_goal: " << settings->goal << std::endl;
    std::cout << "  num_trial: " << settings->r << std::endl;
    std::cout << "  side_len: " << settings->side_len << std::endl;

    std::cout << "step1to4: " << obj.step1to4 << std::endl;
    
    std::cout << "step5_time: [";
    for(int i =0; i < obj.step5_time.size()-1; i++){
        std::cout << obj.step5_time[i] << ",";
    }
    std::cout << obj.step5_time[obj.step5_time.size()-1] << "]" << std::endl;

    std::cout << "step5_x_history_size: [";
    for(int i =0; i < obj.step5_x_history_size.size()-1; i++){
        std::cout << obj.step5_x_history_size[i] << ",";
    }
    std::cout << obj.step5_x_history_size[obj.step5_x_history_size.size()-1] << "]" << std::endl;

    std::cout << "step6a: [";
    for(int i =0; i < obj.step6a.size()-1; i++){
        std::cout << obj.step6a[i] << ",";
    }
    std::cout << obj.step6a[obj.step6a.size()-1] << "]" << std::endl;

    std::cout << "step6b: [";
    for(int i =0; i < obj.step6b.size()-1; i++){
        std::cout << obj.step6b[i] << ",";
    }
    std::cout << obj.step6b[obj.step6b.size()-1] << "]" << std::endl;


    std::cout << "step7: [";
    for(int i =0; i < obj.step7.size()-1; i++){
        std::cout << obj.step7[i] << ",";
    }
    std::cout << obj.step7[obj.step7.size()-1] << "]" << std::endl;

    std::cout << "step8: [";
    for(int i =0; i < obj.step8.size()-1; i++){
        std::cout << obj.step8[i] << ",";
    }
    std::cout << obj.step8[obj.step8.size()-1] << "]" << std::endl;


    std::cout << "step9_time: [";
    for(int i =0; i < obj.step9_time.size()-1; i++){
        std::cout << obj.step9_time[i] << ",";
    }
    std::cout << obj.step9_time[obj.step9_time.size()-1] << "]" << std::endl;

    std::cout << "step9_x_history_size: [";
    for(int i =0; i < obj.step9_x_history_size.size()-1; i++){
        std::cout << obj.step9_x_history_size[i] << ",";
    }
    std::cout << obj.step9_x_history_size[obj.step9_x_history_size.size()-1] << "]" << std::endl;


    std::cout << "step10: [";
    for(int i =0; i < obj.step10.size()-1; i++){
        std::cout << obj.step10[i] << ",";
    }
    std::cout << obj.step10[obj.step10.size()-1] << "]" << std::endl;

    std::cout << "step11: [";
    for(int i =0; i < obj.step11.size()-1; i++){
        std::cout << obj.step11[i] << ",";
    }
    std::cout << obj.step11[obj.step11.size()-1] << "]" << std::endl;
}