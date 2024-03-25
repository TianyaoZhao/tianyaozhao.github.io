# Python语法

## Python基础语法

1. 编码：默认UTF-8编码

2. 标识符

    1. 第一个字符为**字母或下划线**
    2. 标识符由字母数字下划线组成
    3. 大小写敏感

3. 保留字

    ```python
    import keyword
    print(keyword.kwlist)
    ['False', 'None', 'True', 'and', 'as', 'assert', 'break', 'class', 'continue', 'def', 'del', 'elif', 'else', 'except', 'finally', 'for', 'from', 'global', 'if', 'import', 'in', 'is', 'lambda', 'nonlocal', 'not', 'or', 'pass', 'raise', 'return', 'try', 'while', 'with', 'yield']
    ```

4. 注释

    1. 单行注释 `#`

    2. 多行注释

        ```python
        “”“
        多行注释
        ”“”
        
        '''
        多行注释
        '''
        ```

5. 多行语句

    ```python
    total = item_one + \
            item_two + \
            item_three
    ```

    对于`[],{},()`，不需要使用`\`

    ```python
    total = ['item_one', 'item_two', 'item_three',
            'item_four', 'item_five']
    ```

6. 数字类型

    1. int（长整型），python没有long
    2. bool（布尔）
    3. float（浮点） 1.23， 3E-2
    4. complex（复数）1+2j

7. 字符串

    1. `'`和`"`完全相同
    2. `\`用来转义，使用`r`可以防止转义， r"this is a line with \n" ，`\n`会原样输出
    3. `+`连接字符串，`*`运算符重复
    4. 两种索引方式，从左往右`0`开始，从右往左`-1`开始
    5. 字符串的截取：`变量[头下标 : 尾下标 ：步长]`
    6. `"""`和`'''`可以指定多行字符串

8. 空行

    **函数之间或者类的方法之间**用空行分割，表示新的代码开始

9. print输出

    默认换行，如果实现不换行，在变量末尾加上`end=""`

    ```python
    x="a" 
    y="b" 
    # 换行输出 
    print( x ) 
    print( y )  
    print('---------') 
    # 不换行输出 
    print( x, end=" " ) 
    print( y, end=" " ) 
    ```

10. import与from...import

    1. 将整个模块导入

        ```python
        import somemodule
        ```

    2. 从某个模块中导入某个函数

        ```python
        form somemodule import somefunction
        ```

    3. 从某个模块导入多个函数

        ```python
        from somemodule import firstfunc， secondfunc， thirdfunc
        ```

    4. 将某个模块全部函数导入

        ```python
        from somemodule import *
        ```

        

## Python基本数据类型

1. 多个变量赋值

    ```python
    a = b = c = 1
    a, b, c = 1, 2, "runoob"
    ```

2. 标注数据类型

    1. Number

    2. String

    3. bool

    4. List（列表）

    5. Tuple（元组）

    6. Set（集合）

    7. Dictonary（字典）

        不可变数据：Number String Tuple

        可变数据：List Dictonary Set

3. 数值运算

    1. 5 + 4
    2. 4.3 - 2
    3. 3 * 7
    4. 2 / 4 （得到浮点数）
    5. 2 // 4 （得到整数）
    6. 17 % 3 （取余）
    7. 2 ** 5 （乘方）

4. List列表  `[]`

    ```python
    list = [ 'abcd', 786 , 2.23, 'runoob', 70.2 ]
    tinylist = [123, 'runoob']
    
    print (list)            # 输出完整列表
    print (list[0])         # 输出列表第一个元素
    print (list[1:3])       # 从第二个开始输出到第三个元素
    print (list[2:])        # 输出从第三个元素开始的所有元素
    print (tinylist * 2)    # 输出两次列表
    print (list + tinylist) # 连接列表
    
    ['abcd', 786, 2.23, 'runoob', 70.2]
    abcd
    [786, 2.23]
    [2.23, 'runoob', 70.2]
    [123, 'runoob', 123, 'runoob']
    ['abcd', 786, 2.23, 'runoob', 70.2, 123, 'runoob']
    ```



5. Tuple元组 `()`

    **元组和列表一致，但是元组的元素不能修改**

    **可以把字符串看成一种特殊的元组**

    ```python
    #!/usr/bin/python3
    
    tuple = ( 'abcd', 786 , 2.23, 'runoob', 70.2  )
    tinytuple = (123, 'runoob')
    
    print (tuple)             # 输出完整元组
    print (tuple[0])          # 输出元组的第一个元素
    print (tuple[1:3])        # 输出从第二个元素开始到第三个元素
    print (tuple[2:])         # 输出从第三个元素开始的所有元素
    print (tinytuple * 2)     # 输出两次元组
    print (tuple + tinytuple) # 连接元组
    
    ('abcd', 786, 2.23, 'runoob', 70.2)
    abcd
    (786, 2.23)
    (2.23, 'runoob', 70.2)
    (123, 'runoob', 123, 'runoob')
    ('abcd', 786, 2.23, 'runoob', 70.2, 123, 'runoob')
    
    ```

6. Set（集合） `{}`

    Python 中的集合（Set）是一种**无序、可变**的数据类型，用于存储**唯一的元素**。集合中的**元素不会重复**，并且可以进行**交集、并集、差集**等常见的集合操作。

    **注意：**创建一个空集合必须用 **set()** 而不是 **{ }**，因为 **{ }** 是用来创建一个空字典。

    ```python
    #!/usr/bin/python3
    
    sites = {'Google', 'Taobao', 'Runoob', 'Facebook', 'Zhihu', 'Baidu'}
    
    print(sites)   # 输出集合，重复的元素被自动去掉
    
    # 成员测试
    if 'Runoob' in sites :
        print('Runoob 在集合中')
    else :
        print('Runoob 不在集合中')
    
    
    # set可以进行集合运算
    a = set('abracadabra')  
    b = set('alacazam')
    
    print(a)
    
    print(a - b)     # a 和 b 的差集
    
    print(a | b)     # a 和 b 的并集
    
    print(a & b)     # a 和 b 的交集
    
    print(a ^ b)     # a 和 b 中不同时存在的元素
    
    {'Zhihu', 'Baidu', 'Taobao', 'Runoob', 'Google', 'Facebook'}
    Runoob 在集合中
    {'b', 'c', 'a', 'r', 'd'}  已经被拆分了
    {'r', 'b', 'd'}
    {'b', 'c', 'a', 'z', 'm', 'r', 'l', 'd'}
    {'c', 'a'}
    {'z', 'b', 'm', 'r', 'l', 'd'}
    
    ```

    



7. Dictonary（字典）

    列表是有序的对象集合，字典是**无序的对象集合**。两者之间的区别在于：字典当中的元素是**通过键来存取**的，而不是通过偏移存取。

    字典是一种映射类型，字典用 **{ }** 标识，它是一个无序的 **键(key) : 值(value)** 的集合。键(key)必须使用不可变类型。

    在同一个字典中，键(key)必须是唯一的。

    ```python
    #!/usr/bin/python3
    
    dict = {}
    dict['one'] = "1 - 菜鸟教程"
    dict[2]     = "2 - 菜鸟工具"
    
    tinydict = {'name': 'runoob','code':1, 'site': 'www.runoob.com'}
    
    
    print (dict['one'])       # 输出键为 'one' 的值
    print (dict[2])           # 输出键为 2 的值
    print (tinydict)          # 输出完整的字典
    print (tinydict.keys())   # 输出所有键
    print (tinydict.values()) # 输出所有值
    
    
    1 - 菜鸟教程
    2 - 菜鸟工具
    {'name': 'runoob', 'code': 1, 'site': 'www.runoob.com'}
    dict_keys(['name', 'code', 'site'])
    dict_values(['runoob', 1, 'www.runoob.com'])
    
    
    ```

    
