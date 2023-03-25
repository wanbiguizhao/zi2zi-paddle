# zi2zi: Master Chinese Calligraphy with Conditional Adversarial Networks

A zi2zi paddle implement based on [zi2zi-pytorch](https://github.com/xuan-li/zi2zi-pytorch). 

基于[zi2zi-pytorch](https://github.com/xuan-li/zi2zi-pytorch)的paddle版本实现



# 使用说明

## 生成图片
```
python3 font2img.py  --char_size=40 --canvas_size=256 --src_font=data/font/方正隶书_GBK.ttf --dst_font=data/font/方正书宋_GBK.ttf --charset=CN --sample_count=1500 --sample_dir=sample_dir --label=0 --filter --shuffle --mode=font2font
python3 font2img.py  --char_size=40 --canvas_size=64 --src_font=data/font/方正隶书_GBK.ttf --dst_font=data/font/cjk/simhei.ttf --charset=CN --sample_count=1500 --sample_dir=sample_dir --label=1 --filter --shuffle --mode=font2font
python3 font2img.py  --char_size=40 --canvas_size=64 --src_font=data/font/方正隶书_GBK.ttf --dst_font=data/font/cjk/STSONG.TTF --charset=CN --sample_count=1500 --sample_dir=sample_dir --label=2 --filter --shuffle --mode=font2font
python3 font2img.py  --char_size=40 --canvas_size=64 --src_font=data/font/方正隶书_GBK.ttf --dst_font=data/font/ZhongHuaSong/FZSONG_ZhongHuaSongPlane00_2020051520200519101119.TTF --charset=CN --sample_count=1500 --sample_dir=sample_dir --label=3 --filter --shuffle --mode=font2font
python3 font2img.py  --char_size=40 --canvas_size=64 --src_font=data/font/方正隶书_GBK.ttf --dst_font=data/font/cjk/FZSTK.TTF --charset=CN --sample_count=1500 --sample_dir=sample_dir --label=4 --filter --shuffle --mode=font2font

```


```
python3 font2img.py  --char_size=256 --canvas_size=256 --src_font=data/font/方正隶书_GBK.ttf --dst_font=data/font/方正书宋_GBK.ttf --charset=CN --sample_count=1500 --sample_dir=sample_dir --label=0 --filter --shuffle --mode=font2font

python3 font2img.py  --char_size=256 --canvas_size=256 --src_font=data/font/方正隶书_GBK.ttf --dst_font=data/font/cjk/simhei.ttf --charset=CN --sample_count=1500 --sample_dir=sample_dir --label=1 --filter --shuffle --mode=font2font

python3 font2img.py  --char_size=256 --canvas_size=256 --src_font=data/font/方正隶书_GBK.ttf --dst_font=data/font/cjk/STSONG.TTF --charset=CN --sample_count=1500 --sample_dir=sample_dir --label=2 --filter --shuffle --mode=font2font

python3 font2img.py  --char_size=256 --canvas_size=256 --src_font=data/font/方正隶书_GBK.ttf --dst_font=data/font/ZhongHuaSong/FZSONG_ZhongHuaSongPlane00_2020051520200519101119.TTF --charset=CN --sample_count=1500 --sample_dir=sample_dir --label=3 --filter --shuffle --mode=font2font

python3 font2img.py  --char_size=256 --canvas_size=256 --src_font=data/font/方正隶书_GBK.ttf --dst_font=data/font/cjk/FZSTK.TTF --charset=CN --sample_count=1500 --sample_dir=sample_dir --label=4 --filter --shuffle --mode=font2font




python3 font2img.py  --char_size=256 --canvas_size=256 --src_font=data/font/方正兰亭粗黑_GBK.TTF --dst_font=data/font/方正书宋_GBK.ttf --charset=CN --sample_count=1500 --sample_dir=sample_dir --label=0 --filter --shuffle --mode=font2font

```
## 生成二进制训练文件

``` sh
python raw_package.py --dir=sample_dir/  --save_dir=tmp/data   --split_ratio=0.2
```

## 训练模型

```
python train.py --experiment_dir=tmp  --batch_size=128  --input_nc=1 --image_size=64  --epoch=2000 --sample_steps=200 --checkpoint_steps=500




python train.py --experiment_dir=tmp  --batch_size=128  --input_nc=1 --image_size=256  --epoch=10000 --sample_steps=50 --checkpoint_steps=500

```

# 代码结构大致说明




# 预训练模型

# 致谢
