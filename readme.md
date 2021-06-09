# Rasterization Pipeline
- 實做 [Rasterization Pipeline](http://15462.courses.cs.cmu.edu/spring2021/lecture/persp/slide_011)  
![image](https://user-images.githubusercontent.com/49481559/121312247-c75bd280-c937-11eb-98c6-7ff14b3b002e.png)
- [影片說明](https://drive.google.com/file/d/1ExJtAxCEp5sSxON9lHWXPQqKqAUuhhxm/view)

## UI
### Input
#### Camera
- camera 座標 : x, y, z
- frustum : l, r, t, b, n, f
#### Object
- 正四角錐
- 物體中心座標 : x, y, z
- Object Point : 輸入物體中心座標後會自行計算出物體的五個頂點

### Output
#### Coordinate
- 新座標系統為物體放置在 camera 的 -z 方向所建立的
- x, y, z : 新座標系統用原座標系統的 vector 表示
#### Clipping
- 做完 mapping 和 clipping 後的結果，若任一軸座標超出 [-1, 1]，
