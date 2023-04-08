//
//  util.h
//  test_metal
//
//  Created by Владимир Ушаков on 04.01.2023.
//

#ifndef util_h
#define util_h

struct Params {
    float T0;
    float Tb;
    float dt;
    float h;
    float k;
    float dz;

    float kL;
    float kR;
    float kU;
    float kD;
};

struct Flux {
    float left;
    float right;
    float top;
    float bottom;
    float left_edge;
    float right_edge;
    float top_edge;
    float bottom_edge;

    float C;
    float CLgran;
    float CRgran;
    float CUgran;
    float DUgran;
    float CLUgran;
    float CRUgran;
    float CLDgran;
    float DRUgran;
};

#endif /* util_h */
