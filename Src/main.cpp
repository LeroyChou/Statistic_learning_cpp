#include <iostream>
#include "perceptron.h"

using namespace std;




int main() {
    vector<pair<int,int>> x{pair<int, int>(3,3), pair<int, int>(4, 3), pair<int,int>(1,1)};
    vector<int> y {1, 1, -1};
    pair<double, double> w{-2, 2};
    double b{3};
    double lr{0.3};
    Perceptron_dual example{x,y,w,b,lr};
//    Perceptron example{x,y,w,b,lr};
    example.train();
    example.print_results();
//    b = example.get_b();
//    w = example.get_weights();
}
