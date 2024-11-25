#include <iostream>
#include <vector>
#include <omp.h>
#include <random>
#include <algorithm>


int COMPUTATION_TYPE = 0;


class Matrix {
private:
    int size;
    std::vector<std::vector<double>> data;

public:
    explicit Matrix() : size(), data() {}
    explicit Matrix(int size) : size(size), data(size, std::vector<double>(size, 0.0)) {}
    explicit Matrix(int size, double default_value) : size(size), data(size, std::vector<double>(size, default_value)) {}
    Matrix(const Matrix& other) : size(other.size), data(other.data) {}

    ~Matrix() = default;

    Matrix& operator=(const Matrix& other) {
        if (this != &other) {
            size = other.size;
            data = other.data;
        }
        return *this;
    }

    [[nodiscard]] Matrix operator+(const Matrix& other) const {
        if (size != other.size) {
            throw std::invalid_argument("Matrix sizes do not match");
        }
        Matrix result(size);
        for (int i = 0; i < size; ++i) {
            for (int j = 0; j < size; ++j) {
                result.data[i][j] = data[i][j] + other.data[i][j];
            }
        }
        return result;
    }

    void operator+=(const Matrix& other) {
        if (size != other.size) {
            throw std::invalid_argument("Matrix sizes do not match");
        }
        for (int i = 0; i < size; ++i) {
            for (int j = 0; j < size; ++j) {
                data[i][j] += other.data[i][j];
            }
        }
    }

    void randomize() {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(0.0, 1.0);
        for (int i = 0; i < size; ++i) {
            for (int j = 0; j < size; ++j) {
                data[i][j] = dis(gen);
            }
        }
    }

    void randomize_int() {
        for (int i = 0; i < size; ++i) {
            for (int j = 0; j < size; ++j) {
                data[i][j] = rand() % size + 1;
            }
        }
    }

    void set_identity() {
        for (int i = 0; i < size; ++i) {
            for (int j = 0; j < size; ++j) {
                data[i][j] = (i == j) ? 1.0 : 0.0;
            }
        }
    }

    void print() const {
        for (const auto& row : data) {
            for (double val : row) {
                std::cout << val << "\t";
            }
            std::cout << "\n";
        }
    }

    Matrix add(const Matrix& other) {
        if (size != other.size) {
            throw std::invalid_argument("Matrix sizes do not match");
        }

#pragma omp parallel default(none) shared(other, COMPUTATION_TYPE) \
                     num_threads(omp_get_num_threads() <= omp_get_num_procs()/2 \
                                 ? omp_get_num_threads() : omp_get_num_procs()/2)
        if (COMPUTATION_TYPE == 1) {
#pragma omp for
            for (int i = 0; i < size; ++i) {
                for (int j = 0; j < size; ++j) {
                    data[i][j] += other.data[i][j];
                }
            }
        } else {
#pragma omp single
#pragma omp taskgroup
#pragma omp taskloop default(none) shared(other)
            for (int i = 0; i < size; ++i) {
                for (int j = 0; j < size; ++j) {
                    data[i][j] += other.data[i][j];
                }
            }
        }
        return *this;
    }

    [[nodiscard]] Matrix multiply(const Matrix& other) const {
        if (size != other.size) {
            throw std::invalid_argument("Matrix sizes do not match");
        }
        Matrix result(size);

#pragma omp parallel default(none) shared(result, other, COMPUTATION_TYPE)
        if (COMPUTATION_TYPE == 1) {
#pragma omp for
            for (int i = 0; i < size; ++i) {
                for (int j = 0; j < size; ++j) {
                    double sum = 0;
                    for (int k = 0; k < size; ++k) {
                        sum += data[i][k] * other.data[k][j];
                    }
                    result.data[i][j] = sum;
                }
            }
        } else {
#pragma omp single
#pragma omp taskgroup
#pragma omp taskloop default(none) shared(result, other)
            for (int i = 0; i < size; ++i) {
                for (int j = 0; j < size; ++j) {
                    double sum = 0;
                    for (int k = 0; k < size; ++k) {
                        sum += data[i][k] * other.data[k][j];
                    }
                    result.data[i][j] = sum;
                }
            }
        }
        return result;
    }

    [[nodiscard]] Matrix multiply(double number) const {
        Matrix result(size);

#pragma omp parallel default(none) shared(result, number, COMPUTATION_TYPE) \
                     num_threads(omp_get_num_threads() <= omp_get_num_procs()/2 \
                                 ? omp_get_num_threads() : omp_get_num_procs()/2)
        if (COMPUTATION_TYPE == 1) {
#pragma omp for
            for (int i = 0; i < size; ++i) {
                for (int j = 0; j < size; ++j) {
                    result.data[i][j] = number * data[i][j];
                }
            }
        }
        else {
#pragma omp single
#pragma omp taskgroup
#pragma omp taskloop default(none) shared(result, number)
            for (int i = 0; i < size; ++i) {
                for (int j = 0; j < size; ++j) {
                    result.data[i][j] = number * data[i][j];
                }
            }
        }
        return result;
    }

    [[nodiscard]] Matrix exp_matrix(int number) const {
        Matrix result = *this;
        for (int times = 1; times < number; ++times) {
            result = result.multiply(*this);
        }
        return result;
    }

    [[nodiscard]] double trace() const {
        double trace = 0;
        for (int i = 0; i < size; ++i) {
            trace += data[i][i];
        }
        return trace;
    }

    [[nodiscard]] int get_size() const { return size; }
};

Matrix calculate_expression(const Matrix& matrix_B, const Matrix& matrix_C) {
    Matrix matrix_A(matrix_B.get_size());

    // B * C^3
    Matrix C3 = matrix_C.exp_matrix(3);
    matrix_A = matrix_B.multiply(C3);

    // Tr(C) * I
    Matrix matrix_I = Matrix(matrix_C.get_size());
    matrix_I.set_identity();
    matrix_A.add(matrix_I.multiply(matrix_C.trace()));

    // C
    matrix_A.add(matrix_C);

    // Tr(B) * E
    Matrix matrix_E = Matrix(matrix_C.get_size(), 1.0);
    matrix_A.add(matrix_E.multiply(matrix_B.trace()));

    return matrix_A;
}


int main(int argc, char* argv[]) {
    if (argc != 5) {
        std::cerr << \
            "Usage: " << argv[0] << \
            " <matrix_size: int> <threads_count: int> <calculation_type: str> <full_print_mode: bool>" << \
            std::endl;
        return 1;
    }
    int matrix_size = std::stoi(argv[1]);
    if (matrix_size <= 0 || matrix_size > 4096) {
        std::cerr << "Incorrect matrix size.\n";
        return 1;
    }
    int threads_count = std::stoi(argv[2]);
    if (threads_count <= 0) {
        std::cerr << "Incorrect threads count.\n";
        return 1;
    }
    std::string calculation_type = argv[3];
    if (calculation_type == "simple") {
        COMPUTATION_TYPE = 1;
    }
    else if (calculation_type == "tasks"){
        COMPUTATION_TYPE = 2;
    }
    else if (calculation_type == "sections"){
        COMPUTATION_TYPE = 3;
    }
    else {
        std::cerr << "<calculation_type> one of: `simple`, `tasks`, `sections`" << std::endl;
        return 1;
    }
    std::string print_flag = argv[4];
    std::transform(
        print_flag.begin(),
        print_flag.end(),
        print_flag.begin(),
        ::tolower
    );
    bool do_print = (print_flag == "true");

    Matrix matrix_B(matrix_size);
    Matrix matrix_C(matrix_size);

    if (do_print) {
        matrix_B.randomize_int();
        matrix_C.randomize_int();
        std::cout << "Elements of the matrix B:" << std::endl;
        matrix_B.print();
        std::cout << "Elements of the matrix C:" << std::endl;
        matrix_C.print();
    } else {
        matrix_B.randomize();
        matrix_C.randomize();
    }

    omp_set_num_threads(threads_count);
    omp_set_nested(1);
    Matrix matrix_A;
    double Time_1 = 0;
    Time_1 = omp_get_wtime();
    matrix_A = calculate_expression(matrix_B, matrix_C);
    Time_1 = omp_get_wtime() - Time_1;

    if (do_print) {
        std::cout << "Elements of the matrix A:" << std::endl;
        matrix_A.print();
        std::cout << "The time spent on the calculation: " << Time_1 << std::endl;
    }
    else {
        std::cout << Time_1;
    }

    return 0;
}