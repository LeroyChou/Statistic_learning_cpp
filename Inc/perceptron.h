//
// Created by Leroy Chou on 2021/7/13.
//

#ifndef STATISTICS_LEARNING_PERCEPTRON_H
#define STATISTICS_LEARNING_PERCEPTRON_H

#include <utility>
#include <vector>
#include <iostream>
inline std::ostream &
operator << (std::ostream & os, const std::pair<double, double> & w);

class Perceptron{
private:
    std::vector<std::pair<int, int>> x;
    std::vector<int> y;
    std::pair<double, double> w;
    double b;
    double lr;

public:
    Perceptron(std::vector<std::pair<int, int>> &  x, std::vector<int> &  y, std::pair<double, double> & w,
            double& b, double& lr)
            :x(x), y(y), w(w), b(b), lr(lr){
        std::cout << "Weights get initialized: " << w << "\n";
        std::cout << "Intercept get initialized: " << b << "\n";

    }

    void train(){
        std::cout << "\nTraining..." << "\n\n";
        int cout=0;
        while (true){
            bool flag = true;
            for(auto i = 0; i < x.size(); ++i){
                if ((w.first*x.at(i).first + w.second*x.at(i).second+b) * y.at(i) <= 0){
                    flag = false;
                    w.first += lr * y.at(i) * x.at(i).first;
                    w.second += lr * y.at(i) * x.at(i).second;
                    b += lr * y.at(i);
                    --i;
                    std::cout << "Weights get updated: " << w << "\n";
                    std::cout << "Intercept get updated: " << b << "\n\n";
                    std::cout << ++cout << std::endl;

                }
            }
            if(flag){
                std::cout << "Training Done!" << "\n";
                break;
            }
        }
    }

    void print_results(){
        std::cout << "\nThe final trained weights: " << w << std::endl;
        std::cout << "The final trained intercept: " << b << std::endl;
    }

};

inline std::ostream &
operator << (std::ostream & os, const std::pair<double, double> & w) {
    os << "(" << w.first << "," << w.second << ")";
    return os;
}

class Perceptron_dual{
private:
    std::vector<std::pair<int, int>> x;
    std::vector<int> y;
    std::pair<double, double> w;
    double b;
    double lr;
    std::vector<double> alpha;
    int** Gram;
public:
    Perceptron_dual(std::vector<std::pair<int, int>> &  x, std::vector<int> &  y, std::pair<double, double> & w,
    double& b, double& lr)
    :x(x), y(y), w(w), b(b), lr(lr) {
        std::cout << "Weights get initialized: " << w << "\n";
        std::cout << "Intercept get initialized: " << b << "\n";
        Gram = new int*[x.size()];
        for (auto i=0; i<x.size();++i) {
            Gram[i] = new int[x.size()];
        }
        for (auto i=0; i<x.size();++i){
            for (auto j=0; j<x.size();++j) {
                Gram[i][j] = x.at(i).first*x.at(j).first + x.at(i).second*x.at(j).second;
            }
        }
        alpha.insert(alpha.begin(), x.size(), 0);
    }
    ~Perceptron_dual(){
        for (auto i=0; i<x.size();++i) {
            delete [] Gram[i];
        }
        delete [] Gram;
    }
    void train(){
        std::cout << "\nTraining..." << "\n\n";
        int cout = 0;

        while(true){
            bool flag = true;
            for(auto i = 0; i < x.size(); ++i){
                double temp = 0;
                for(auto j = 0; j < x.size(); ++j){
                    temp += alpha.at(j) * y.at(j) * Gram[j][i] + alpha.at(j)*y.at(j);
                }
                temp += x.at(i).first * w.first + x.at(i).second * w.second + b;
                if (temp * y.at(i) <= 0){
                    flag = false;
                    alpha.at(i) += lr;
                    std::cout << "Alpha of x" << i+1 << " get updated: " << alpha.at(i) << std::endl;
                    --i;
                    std::cout << ++cout << std::endl;

                }
            }
            if (flag) {
                std::cout << "\nTraining Done! Trained Alpha is ("  << alpha.at(0)<<", " << alpha.at(1) <<", "<< alpha.at(2)
                << ")"<< std::endl;
                break;
            }
        }
    }
    void print_results(){
        for (auto i=0; i<x.size();++i){
            w.first += alpha.at(i)*y.at(i)*x.at(i).first;
            w.second += alpha.at(i)*y.at(i)*x.at(i).second;
            b += alpha.at(i) * y.at(i);
        }
        std::cout << "\nThe final trained weights: " << w << std::endl;
        std::cout << "The final trained intercept: " << b << std::endl;
    }
};



#endif //STATISTICS_LEARNING_PERCEPTRON_H
