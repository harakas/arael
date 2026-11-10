/* The vendored Python math types by value through C (python.rs): each
 * function returns its argument doubled, so the value crosses both ways.
 * The layouts are arael's repr(C) vect<T, N> / matrix<T, R, C>. */
typedef struct { double e[4]; } vectnd4;
typedef struct { float e[4]; } vectnf4;
typedef struct { double x, y; } vect2d;
typedef struct { vect2d r[2]; } matrix2d;
typedef struct { vectnd4 r[2]; } matrixnd24;
typedef struct { double e[2]; } vectnd2;
typedef struct { vectnd2 r[2]; } matrixnd22;
typedef struct { double x, y, z; } vect3d;
typedef struct { vect3d r[3]; } matrix3d;

vectnd4 twice_vectnd4(vectnd4 v) { for (int i = 0; i < 4; i++) v.e[i] *= 2; return v; }
vectnf4 twice_vectnf4(vectnf4 v) { for (int i = 0; i < 4; i++) v.e[i] *= 2; return v; }
vect2d twice_vect2d(vect2d v) { v.x *= 2; v.y *= 2; return v; }
matrix2d twice_matrix2d(matrix2d m) {
    for (int i = 0; i < 2; i++) { m.r[i].x *= 2; m.r[i].y *= 2; }
    return m;
}
matrixnd22 twice_matrixnd22(matrixnd22 m) {
    for (int i = 0; i < 2; i++) for (int j = 0; j < 2; j++) m.r[i].e[j] *= 2;
    return m;
}
matrixnd24 twice_matrixnd24(matrixnd24 m) {
    for (int i = 0; i < 2; i++) for (int j = 0; j < 4; j++) m.r[i].e[j] *= 2;
    return m;
}
matrix3d twice_matrix3d(matrix3d m) {
    for (int i = 0; i < 3; i++) { m.r[i].x *= 2; m.r[i].y *= 2; m.r[i].z *= 2; }
    return m;
}
