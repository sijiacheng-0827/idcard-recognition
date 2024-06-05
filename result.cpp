#include <stdio.h>

typedef struct {
    int min;
    int max;
} Range;

typedef struct {
    int top;
    int left;
} Position;



typedef enum {
    AREA_0 = 0,
    AREA_1 = 1,
    AREA_2 = 2,
    AREA_3 = 3,
    AREA_4 = 4,
    AREA_5 = 5,
    AREA_6 = 6,
    AREA_7 = 7,
    AREA_8 = 8,
    AREA_9 = 9,
    AREA_10 = 10,
    AREA_11 = 11
} IdCardDirtyArea;

typedef enum {
    NOT_DIRTY = 0,
    DIRTY = 1
} DirtyState;

int idcard_dirty_detect(const int front,  int* result) {
    // 模拟检测算法
    Range ranges_front[7] = { {0, 59}, {60, 115}, {116, 339}, {340, 419}, {420, 639}, {60, 475}, {415, 599} };
    Range ranges_back[4] = { {0, 299}, {415, 496}, {497, 639}, {0, 299} };
    Position position = { -1, -1 };  // 假设的位置信息，实际使用时需根据图像识别结果赋值

    // 根据位置信息和卡片方向判断所在区域
    
    int area = -1;
    if (front == 1) {
        if (position.top >= ranges_front[0].min && position.top <= ranges_front[0].max) {
            area = AREA_0;
        }
        else if (position.top >= ranges_front[1].min && position.top <= ranges_front[1].max) {
            if (position.left >= 0 && position.left <= 173) {
                area = AREA_1;
            }
            else if (position.left > 173 && position.left <= 339) {
                area = AREA_2;
            }
        }
        else if (position.top >= ranges_front[2].min && position.top <= ranges_front[2].max && position.left >= 0 && position.left <= 380) {
            area = AREA_3;
        }
        else if (position.top >= ranges_front[3].min && position.top <= ranges_front[3].max && position.left >= 0 && position.left <= 420) {
            area = AREA_4;
        }
        else if (position.top >= ranges_front[4].min && position.top <= ranges_front[4].max && position.left >= 0 && position.left <= 640) {
            area = AREA_5;
        }
        else if (position.top >= ranges_front[5].min && position.top <= ranges_front[5].max && position.left >= 415 && position.left <= 600) {
            area = AREA_6;
        }
    }
    else {
        if (position.top >= ranges_back[0].min && position.top < ranges_back[0].max && position.left >= 0 && position.left < 150) {
            area = AREA_7;
        }
        else if (position.top >= ranges_back[1].min && position.top <= ranges_back[1].max && position.left >= 0 && position.left < 640) {
            area = AREA_8;
        }
        else if (position.top >= ranges_back[2].min && position.top < ranges_back[2].max && position.left >= 0 && position.left < 640) {
            area = AREA_9;
        }
        else if (position.top >= ranges_back[3].min && position.top < ranges_back[3].max && position.left >= 150 && position.left < 640) {
            area = AREA_11;
        }
    }

    // 根据区域判断结果
    if (area != -1) {
        result[area] = DIRTY;
    }

    return 0; // 返回执行成功
}

int main() {
    const int front = 0;
    
    int results[12] = { NOT_DIRTY };

    int ret = idcard_dirty_detect(front, results);

    if (ret == 0) {
        printf("Detection succeeded\n");
        for (int i = 0; i < 12; ++i) {
            printf("Area %d: %s\n", i, results[i] == DIRTY ? "Dirty" : "Not dirty");
        }
    }
    else {
        printf("ok\n");
    }

    return 0;
}