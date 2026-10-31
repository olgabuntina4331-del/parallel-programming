#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <algorithm>
#include <limits>
#include <omp.h>
#include <string>

//точка (x, y)
struct Point {double x, y;};

//читаем точки
bool readPoints(const std::string& filename, std::vector<Point>& points) {
    std::ifstream file(filename);
    int n;
    if (!(file >> n) || n <= 0) {
        std::cerr << "Error: cannot read point count from " << filename << '\n';
        return false;
    }
    points.resize(n);
    for (int i = 0; i < n; ++i) {
        if (!(file >> points[i].x >> points[i].y) || !std::isfinite(points[i].x) || !std::isfinite(points[i].y)) {
            std::cerr << "Error: invalid point " << i + 1 << '\n';
            return false;
        }
    }
    return true;
}

//суммы для нормальных уравнений
void calculateSums(const std::vector<Point>& points,double (&sumX)[7], double (&sumXY)[4]) {
    int n = static_cast<int>(points.size());

    //каждый поток собирает свои суммы, потом они объединяются
    #pragma omp parallel for reduction(+:sumX[:7],sumXY[:4])
    for (int i = 0; i < n; ++i) {
        double power = 1.0;
        for (int k = 0; k <= 6; ++k) {
            sumX[k] += power;
            if (k <= 3) sumXY[k] += points[i].y * power;
            power *= points[i].x;
        }
    }
}

//решаем систему гауссом с выбором ведущей строки
bool gaussianElimination(std::vector<std::vector<double>> A, std::vector<double> b, std::vector<double>& coefficients) {
    int n = static_cast<int>(b.size());
    for (int k = 0; k < n; ++k) {
        int pivot = k;
        for (int i = k + 1; i < n; ++i) {
            if (std::fabs(A[i][k]) > std::fabs(A[pivot][k])) pivot = i;
        }
        if (std::fabs(A[pivot][k]) < 1e-12) return false;
        std::swap(A[k], A[pivot]);
        std::swap(b[k], b[pivot]);

        for (int i = k + 1; i < n; ++i) {
            double factor = A[i][k] / A[k][k];
            for (int j = k; j < n; ++j) A[i][j] -= factor * A[k][j];
            b[i] -= factor * b[k];
        }
    }

    coefficients.assign(n, 0.0);
    for (int i = n - 1; i >= 0; --i) {
        double value = b[i];
        for (int j = i + 1; j < n; ++j) value -= A[i][j] * coefficients[j];
        coefficients[i] = value / A[i][i];
    }
    return true;
}

//полином по схеме горнера
double polynomial(double x, const std::vector<double>& a) {
    double y = 0.0;
    for (int j = static_cast<int>(a.size()) - 1; j >= 0; --j) {
        y = y * x + a[j];
    }
    return y;
}

//сумма квадратов ошибок, параллельный расчёт
double squaredError(const std::vector<Point>& points, const std::vector<double>& a) {
    double error = 0.0;
    int n = static_cast<int>(points.size());
    #pragma omp parallel for reduction(+:error)
    for (int i = 0; i < n; ++i) {
        double diff = points[i].y - polynomial(points[i].x, a);
        error += diff * diff;
    }
    return error;
}

void printPolynomial(const std::vector<double>& a) {
    std::cout << "P(x) = " << a[0];
    for (int j = 1; j < static_cast<int>(a.size()); ++j) {
        if (std::fabs(a[j]) < 1e-8) continue;
        std::cout << (a[j] < 0 ? " - " : " + ") << std::fabs(a[j]) << "*x";
        if (j > 1) std::cout << "^" << j;
    }
    std::cout << '\n';
}

int main() {
    std::vector<Point> points;
    if (!readPoints("points_100000.txt", points)) return 1;

    double sumX[7] = {};
    double sumXY[4] = {};

    //начало вычислений
    double startTime = omp_get_wtime();

    //суммы для сист. норм. ур-й
    calculateSums(points, sumX, sumXY);

    //коэффициентов меньше, чем есть точек
    int maxDegree = std::min(3, static_cast<int>(points.size()) - 1);
    double yScale = 0.0;
    for (const Point& p : points) yScale += p.y * p.y;
    double tolerance = 1e-10 * std::max(1.0, yScale);

    double bestError = std::numeric_limits<double>::infinity();
    int bestDegree = -1;
    std::vector<double> bestCoefficients;

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Points: " << points.size() << '\n';
    std::cout << "OpenMP max threads: " << omp_get_max_threads() << "\n\n";

    for (int degree = 0; degree <= maxDegree; ++degree) {
        int size = degree + 1;
        std::vector<std::vector<double>> A(size, std::vector<double>(size));
        std::vector<double> b(size);

        //сист. норм. ур-й
        for (int i = 0; i < size; ++i) {
            b[i] = sumXY[i];
            for (int j = 0; j < size; ++j) A[i][j] = sumX[i + j];
        }

        std::vector<double> a;
        if (!gaussianElimination(A, b, a)) {
            std::cout << "Degree " << degree << ": singular system\n\n";
            continue;
        }

        double error = squaredError(points, a);
        std::cout << "Degree " << degree << ": ";
        printPolynomial(a);
        std::cout << "SSE = " << error << "\n\n";

        //если ошибки практически равны, оставляем меньшую степень
        if (bestDegree == -1 || error < bestError - tolerance) {
            bestError = error;
            bestDegree = degree;
            bestCoefficients = a;
        }
    }

    //время выполнения
    double elapsed = omp_get_wtime() - startTime;

    std::cout << "Calculation time (seconds): " << std::setprecision(9) << elapsed << std::setprecision(6) << "\n\n";
    
    if (bestDegree == -1) {
        std::cerr << "Error: no polynomial found\n";
        return 1;
    }

    std::cout << "BEST POLYNOMIAL (degree " << bestDegree << "):\n";
    printPolynomial(bestCoefficients);
    std::cout << "SSE = " << bestError << "\n\n";

    //выводим первые 10 точек
    int count = std::min(10, static_cast<int>(points.size()));

    std::cout << "\nComparison (first " << count << " of " << points.size() << " points):\n";

    //исходное и рассчитанное значение
    for (int i = 0; i < count; ++i) {
        const Point& p = points[i];
        std::cout << p.x << "  " << p.y << "  " << polynomial(p.x, bestCoefficients) << '\n';
    }

    //сохраняем коэффициенты, чтобы построить график
    std::ofstream result("result.txt");
    if (!result) {
        std::cerr << "Error: cannot create result.txt\n";
        return 1;
    }
    result << bestDegree << '\n' << std::setprecision(17);
    for (double a : bestCoefficients) result << a << ' ';
    result << '\n' << bestError << '\n';
    return 0;
}
