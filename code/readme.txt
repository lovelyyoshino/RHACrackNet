1. 将对应数据放置到data文件夹内
2. 训练方式： python main.py  ./config/unet_CamCrack789.yaml
3. 所有训练参数：config文件夹内对应修改
4. 替换不同模型：修改config文件夹内文件的model项目，例如：depthwise 使用unet_dep

说明：
models文件里面的unet是指论文所提出的RHACrackNet，unet-dep是指RHACrackNet*（采用Depth-separable convolution代替传统卷积）