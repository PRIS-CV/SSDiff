# color_transfer


## Requirements
- OpenCV
- NumPy

## Results
![res_a](docs/res_a.jpg?raw=true)
![res_b](docs/res_b.jpg?raw=true)



## How to use
```python
python color_transfer.py -o (path of your origin image) -r (path of your reference image) -g (path of your generated image)
```
for example:
```py
python color_transfer.py -o ./images/a.jpg -r ./images/b.jpg -g new.jpg
```