# 问题

如果使用`allennlp train ...`来训练模型，那如何来调试代码呢？

# 解决方案

那我们就使用`python`来模仿命令的执行，这样就可以debug代码了。

首先添加`debug.py`文件。

内容如下所示：

```python
import json
import shutil
import sys

from allennlp.commands import main

config_file = "/path/to/config.json"

# 在cuda上训练模型.
overrides = json.dumps({"trainer": {"cuda_device": -1}})

serialization_dir = "./output"

# allennlp会检测serialization_dir 文件夹是否存在，如果存在就会报错
# 于是每次重新运行代码的时候，都需要手动删除这个文件夹
shutil.rmtree(serialization_dir, ignore_errors=True)

# 手动指定命令行参数
sys.argv = [
    "allennlp",  # command name, not used by main
    "train",
    config_file,
    "-s", serialization_dir,
    "--include-package", "my_library",
    "-o", overrides,
]

main()
```