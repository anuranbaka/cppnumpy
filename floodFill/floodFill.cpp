#include <floodFill.h>

bool fuzzyFill1(int current_color, int target_color){
    if(current_color >= target_color - 1 && current_color <= target_color + 1) return true;
    else return false;
}
bool fuzzyFill3(int current_color, int target_color){
    if(current_color >= target_color - 3 && current_color <= target_color + 3) return true;
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
    Mat<int> map3({0,0,0,0,9,0,0,0,0,
                   0,0,0,0,9,1,1,1,0,
                   0,0,0,0,9,0,0,1,0,
                   0,0,0,0,9,0,0,1,0,
                   9,9,9,9,9,9,9,9,9,
                   0,1,0,0,9,0,0,0,0,
                   0,1,0,0,9,0,0,0,0,
                   0,1,1,1,9,0,0,0,0,
                   0,0,0,0,9,0,0,0,0},9,9);
    FILE* outFile;
    outFile = fopen("floodFill/floodOutput.txt","w");

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

    fprintf(outFile,"Map2 before flood fill:\n");
    map2.print(outFile);
    
    start = {4,4};
    floodFillCustom(map2, start, 5, fuzzyFill1);
    fprintf(outFile,"Map2 Custom filling center with 5 (threshold 1):\n");
    map2.print(outFile);

    start = {4,4};
    floodFillCustom(map2, start, 0, fuzzyFill1);
    start = {4,4};
    floodFillCustom(map2, start, 5, fuzzyFill3);
    fprintf(outFile,"Map2 reset, then Custom filling center with 5 (threshold 3):\n");
    map2.print(outFile);

    fprintf(outFile,"Map3 before flood fill:\n");
    map3.print(outFile);
    
    start = {7,2};
    floodFillCustom(map3, start, 5, fuzzyFill1);
    fprintf(outFile,"Map3 Custom filling lower left corner with 5 (threshold 1):\n");
    map3.print(outFile);

    start = {0,6};
    Mat<int> subMat = map3.roi(7,8);
    floodFill(subMat, start, 5);
    fprintf(outFile,"Map3 fill on row 7 starting from column 7:\n");
    map3.print(outFile);

    return 0;
}
