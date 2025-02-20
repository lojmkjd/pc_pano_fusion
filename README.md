# pc_pano_fusion

```mermaid
graph TD
    %% 主要数据流
    Start([开始]) --> Init[初始化 DataProcessor]
    Init --> LoadData{加载数据}

%% 数据加载和验证
subgraph DataLoading[数据加载与验证]
    LoadData --> |验证成功| DataReady[数据就绪]
    LoadData --> |验证失败| ErrorHandler[错误处理]
    ErrorHandler --> Exit([结束])
end

%% 并行处理模块
DataReady --> ParallelProcess

subgraph ParallelProcess[并行处理模块]
    direction TB
    
    subgraph PointCloud[点云处理]
        PC1[读取点云] --> PC2[下采样]
        PC2 --> PC3[去除离群点]
        PC3 --> PC4[坐标转换]
    end

​    subgraph Panorama[全景图处理]
​        P1[加载图像] --> P2[图像预处理]
​    end
end

%% 融合处理
ParallelProcess --> Fusion

subgraph Fusion[融合处理]
    direction TB
    F1[坐标转换] --> F2[像素映射]
    F2 --> F3[颜色提取]
    F3 --> F4[颜色融合]
end

%% 优化和输出
Fusion --> Optimization

subgraph Optimization[优化与输出]
    direction TB
    O1[计算平均颜色] --> O2[颜色校正]
    O2 --> O3[生成彩色点云]
    O3 --> O4[结果可视化]
end

%% 结果保存
Optimization --> SaveResults[保存结果]
SaveResults --> Success([处理完成])

%% 错误处理流
LoadData --> |数据异常| ErrorHandler
ParallelProcess --> |处理异常| ErrorHandler
Fusion --> |融合异常| ErrorHandler
Optimization --> |优化异常| ErrorHandler

%% 样式定义
classDef default fill:#f9f9f9,stroke:#333,stroke-width:1px;
classDef process fill:#ddf1d5,stroke:#82b366,stroke-width:2px;
classDef error fill:#f8cecc,stroke:#b85450,stroke-width:2px;
classDef success fill:#d5e8d4,stroke:#82b366,stroke-width:2px;
classDef module fill:#dae8fc,stroke:#6c8ebf,stroke-width:2px;

%% 应用样式
class Start,Success success;
class ErrorHandler error;
class DataLoading,ParallelProcess,Fusion,Optimization module;
class LoadData,SaveResults process;
```

