#include <iostream>
#include <string>
#include "solver_trans_intra_2ac.h"

// Main
int main(int argc, char ** argv) {
    int N = 1000;
    if (argc >= 2)
       N = std::atoi(argv[1]);
    solver_trans_intra_2ac(N);

	return 0;
}