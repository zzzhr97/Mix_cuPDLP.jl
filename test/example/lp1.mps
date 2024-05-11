* Min  2x - 1.4y + 4.2z - 6t
* s.t. x + y + z + t <= 14
*      5x - 2.2y + 4.9z + 11.8t = 8.2
*      0 <= x
*      0 <= y
*      0 <= z
*      0 <= t <= 11
NAME trivial_lp_model
ROWS
 N  OBJ
 L  con1
 E  con2
COLUMNS
     x        con1        1
     x        con2      5
     x        OBJ       2
     y        con1      1
     y        con2      -2.2
     y        OBJ       -1.4
     z        con1      1
     z        con2      4.9
     z        OBJ       4.2
     t        con1      1
     t        con2      11.8
     t        OBJ       -6
RHS
    rhs       con1      14
    rhs       con2      8.2
RANGES
BOUNDS
 LO bounds    x        0
 LO bounds    y        0
 LO bounds    z        0
 LO bounds    t        0
 UP bounds    t        11
ENDATA