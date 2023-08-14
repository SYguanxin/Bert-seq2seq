写爬虫ing


## 之前考核写过了，再写一遍也毫不费力(咬牙，所以爬虫没什么学习笔记

#### 但是之前只用数据库存数据，这次打算用json
### 对json文件追加内容 
> json {"key_1": "value_1"}
```python
import json

new_data = {"key_2": "value_2"}

with open("test.json", "r", encoding="utf-8") as f:
    old_data = json.load(f)
    old_data.update(new_data)
with open("test.json", "w", encoding="utf-8") as f:
    json.dump(old_data, f)
```
### 修改json文件
```python
import json

with open("test.json", "r", encoding="utf-8") as f:
    old_data = json.load(f)
    old_data["key_2"] = "value_3"
with open("test.json", "w", encoding="utf-8") as f:
    json.dump(old_data, f)
```
> 写入json.dump时需要设置ensure_ascii=False,否则是unicode编码