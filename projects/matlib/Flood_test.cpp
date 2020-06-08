#include "floodFill.h"
#include <vector>
using namespace std;

bool fuzzyFill(int current_color, int target_color){
    if(current_color >= target_color - 1 && target_color <= target_color + 1) return true;
    else return false;
}

int main(){
    Mat<int> map1({9,9,9,9,9,9,9,9,9,
                   9,0,0,0,9,0,0,0,9,
                   9,0,0,0,9,0,0,0,9,
                   9,0,0,9,0,0,0,0,9,
                   9,9,9,0,0,0,9,9,9,
                   9,0,0,0,0,9,0,0,9,
                   9,0,0,0,9,0,0,0,9,
                   9,0,0,0,9,0,0,0,9,
                   9,9,9,9,9,9,9,9,9},9,9);
    Mat<int> map2({9,9,9,9,9,9,9,9,9,
                   9,0,0,0,9,1,1,1,9,
                   9,0,0,0,2,0,0,1,9,
                   9,0,0,2,0,0,0,1,9,
                   9,9,2,0,0,0,2,9,9,
                   9,0,0,0,0,2,0,0,9,
                   9,1,0,0,2,0,0,0,9,
                   9,1,1,1,9,0,0,0,9,
                   9,9,9,9,9,9,9,9,9},9,9);
    FILE* outFile;
    outFile = fopen("floodOutput.txt","w");

    fprintf(outFile,"Map1 before flood fill:\n");
    map1.print(outFile);
    
    vector<size_t> start = {2,2};
    floodFill(map1, start, 5, 8);

    fprintf(outFile,"Filling top-left with 5 (connectivity 8):\n");
    map1.print(outFile);

    start = {4,4};
    floodFill(map1, start, 7);

    fprintf(outFile,"Filling center with 7 (connectivity 4):\n");
    map1.print(outFile);
    
    return 0;
}