# JS的调用方式与执行顺序

**使用方式**

HTML页面中的任意位置加上`<script type="module"></script>`标签即可

常见使用方式有以下几种：

- 直接在`<script type="module"></script>`标签内写JS代码

- 直接引入文件：`<script type="module" src="/static/js/index.js"></script>`

- 将所需的代码通过`import`关键字引入到当前作用域

`/static/js/index.js`文件中的内容为：

```js
let name = "acwing";

function print() {
    console.log("Hello World!");
}
// 将变量或者是函数导出到外部
export {
    name,
    print
}
```

`<script type="module"></script>`中的内容为：

```html
<script type="module">
    <!-- 将需要的参量导入 -->
    import { name, print } from "/static/js/index.js";

    console.log(name);
    print();
</script>
<div></div>
<input></input>
```

**执行顺序**

- 类似于HTML与CSS，按从上到下的顺序执行
- 事件驱动执行

**HTML, CSS, JavaScript三者之间的关系**

- CSS控制HTML
- JavaScript控制HTML与CSS
- 为了方便开发与维护，尽量按照上述顺序写代码。例如：不要在HTML中调用JavaScript中的函数

# 变量与运算符

# 输入与输出

# 判断语句

# 循环语句

# 对象

# 数组

# 函数

# 类

# 事件

JavaScript的代码一般通过事件触发。

可以通过`addEventListener`函数为元素绑定事件的触发函数。

常见的触发函数有：

```js
 // 比如我想控制一个div组件
let div = document.querySelevtor('div');

let main() = function(){
    div.addEventListener('mousedown', function(event){
       //具体的事件触发逻辑
        console.log(event.type, event.button);
    });
}
// 导出函数或者是
export{
	main
}
```

**鼠标**

- `click`：鼠标左键点击（按下和弹起）
- `dblclick`：鼠标左键双击
- `contextmenu`：鼠标右键点击
- `mousedown`：鼠标按下，包括左键、滚轮、右键  按下都可以触发
  - `event.button`：返回 0表示左键，1表示中键，2表示右键
- `mouseup`：鼠标弹起，包括左键、滚轮、右键 弹起都可以触发
  - `event.button`：返回  0表示左键，1表示中键，2表示右键

**键盘**

- `keydown`：某个键是否被按住，事件会连续触发
  - `event.code`：返回按的是哪个键
  - `event.altKey、event.ctrlKey、event.shiftKey`分别表示是否同时按下了`alt、ctrl、shift`键。
- `keyup`：某个按键是否被释放
  - `event`常用属性同上
- `keypress`：紧跟在`keydown`事件后触发，只有按下字符键时触发。适用于判定用户输入的字符。
  - `event`常用属性同上
- `keydown、keyup、keypress`的关系类似于鼠标的`mousedown、mouseup、click`

**表单**

- `focus`：聚焦某个元素
- `blur`：取消聚焦某个元素
- `change`：某个元素的内容发生了改变常用库

**窗口**

需要作用到`window`元素上。

- `resize`：当窗口大小放生变化
- `scroll`：滚动指定的元素
- `load`：当元素被加载完成



# 常用库

## jQuery



更方便的去获取前端的标签

**使用方式**

- 在<head>元素中添加：

  ```html
  <script src="https://cdn.acwing.com/static/jquery/js/jquery-3.3.1.min.js"></script>
  ```

- 按jQuery官网提示下载

新写法

```js
let main = function(){
    // $在变量中就可以当成字符来处理
    let $div = $('div');
    console.log($div);
}
export{
	main
}
```

**选择器**

`$(selector)`，例如：

```js
$('div'); 		//选择标签
$('.big-div');	//选择类
$('#big-div');	//选择id
$('div > p')    //选择子节点p
$('div   p')    //选择所有后代p
```

`selector`类似于CSS选择器。

**事件**

`$(selector).on(event, func)`绑定事件，例如：

```js
$('div').on('click', function (event) {
    console.log("click div");
})
```

`$(selector).off(event, func)`删除事件，例如：

```js
$('div').on('click', function (e) {
    console.log("click div");

    $('div').off('click');
});
```

当存在多个相同类型的事件触发函数时，可以通过`click.name`来区分，例如：

```js
$('div').on('click.first', function (e) {
    console.log("click div");

    $('div').off('click.first'); // 解绑掉click.first
});
```

在事件触发的函数中的`return false`等价于同时执行：

- `event.stopPropagation()`：阻止事件向上传递（子标签执行完毕，父标签也会执行）

- `event.preventDefault()`：阻止当前点击事件的默认行为，并不阻止向上传递（子标签不会执行，父标签会执行）

**==A都是标签==**

**元素的隐藏、展现**

- `$A.hide()`：隐藏，可以添加参数，表示消失时间
- `$A.show()`：展现，可以添加参数，表示出现时间
- `$A.fadeOut()`：慢慢消失，可以添加参数，表示消失时间
- `$A.fadeIn()`：慢慢出现，可以添加参数，表示出现时间

**元素的添加、删除**

- `$('<div class="mydiv"><span>Hello World</span></div>')`：构造一个jQuery对象(构造一个标签，可以任意嵌套，跟写html一样)
- `$A.append($B)`：将`$B`添加到`$A`的末尾
- `$A.prepend($B)`：将`$B`添加到`$A`的开头
- `$A.remove()`：删除元素`$A`
- `$A.empty()`：清空元素`$A`的所有儿子

**对类的操作**

- `$A.addClass(class_name)`：添加某个类
- `$A.removeClass(class_name)`：删除某个类
- `$A.hasClass(class_name)`：判断某个类是否存在

**对CSS的操作**

- `$("div").css("background-color")`：获取某个标签的CSS的属性

- `$("div").css("background-color","yellow")`：设置某个CSS的属性

- 同时设置多个CSS的属性：

 ```js
  $('div').css({
      width: "200px",
      height: "200px",
      "background-color": "orange",
  });
 ```

**对标签属性的操作**

`id`可以任意指定任意命名

- `$('div').attr('id')`：获取属性
- `$('div').attr('id', 'ID')`：设置属性



**对HTML内容、文本的操作**

不需要背每个标签该用哪种，用到的时候Google或者百度即可。

- `$A.html()`：获取、修改HTML内容(完整的标签＋内容)
- `$A.text()`：获取、修改文本信息（只有文本内容）
- `$A.val()`：获取、修改文本的值（input常用）



**查找**

`filter`一般是类选择器形式

- `$(selector).parent(filter)`：查找父元素
- `$(selector).parents(filter)`：查找所有祖先元素
- `$(selector).children(filter)`：在所有子元素中查找
- `$(selector).find(filter)`：在所有后代元素中查找（用的最多）



==ajax==

跟后端进行通信，不刷新页面的情况下，只获取后端的某些数据

**GET方法 **    从服务器端获取数据

```js
$.ajax({
    url: url,     //后端的链接
    type: "GET",
    data: {		  //往后端传的各种参数
    },
    dataType: "json",
    success: function (resp) {  // 后端传回的信息，从resp中解析出来

    },
});
```

**POST方法**    表单form提交数据

```js
$.ajax({
    url: url,
    type: "POST",
    data: {
    },
    dataType: "json",
    success: function (resp) {

    },
});
```



## setTimeout与setInterval



`setTimeout(func, delay)`

`delay`毫秒后，执行函数`func()`

 

`clearTimeout()`

关闭定时器，例如：

```JS
let main() = function(){
    let $div = $('div');
    
    $div.click(function(){
        	setTimeout(function(){
            	console.log("hello")
        }, 2000);    //2s后控制台输出hello
    });
}
```

`setInterval(func, delay)`

每隔`delay`毫秒，执行一次函数`func()`
第一次在第delay毫秒后执行。

`clearInterval()`



## requestAnimationFrame

做动画的函数

## Map与Set

`Map`

Map 对象保存键值对

- 用`for...of`或者`forEach`可以按插入顺序遍历。
- 键值可以为任意值，包括函数、对象或任意基本类型。

常用API：

- `set(key, value)`：插入键值对，如果`key`已存在，则会覆盖原有的`value`
- `get(key)`：查找关键字，如果不存在，返回`undefined`
- `size：返回键值对数量`
- `has(key)`：返回是否包含关键字`key`
- `delete(key)`：删除关键字`key`
- `clear()`：删除所有元素

```js
let mian = function(){
    let map = new Map();
    map.set('name', 'yxc');
    map.set('age', 18);
    
    // 遍历
    for (let [key, value] of map){
        console.log(key, value);
    }
}
```

`Set`
Set 对象允许你存储任何类型的唯一值，无论是原始值或者是对象引用。

用`for...of`或者`forEach`可以按插入顺序遍历。
常用API：

- `add()`：添加元素
- `has()`：返回是否包含某个元素
- `size`：返回元素数量
- `delete()`：删除某个元素
- `clear()`：删除所有元素

```js
let mian = function(){
    let set = new Set();
    map.add('name');
    map.set(18);
    
    // 遍历
    for (let id of set){
        console.log(id);
    }
}
```



## localStorage



可以在用户的浏览器上存储键值对。

常用API：

- `setItem(key, value)`：插入
- `getItem(key)`：查找
- `removeItem(key)`：删除
- `clear()`：清空

## JSON

JSON对象用于序列化对象、数组、数值、字符串、布尔值和`null`。

常用API：

- `JSON.parse()`：将字符串解析成对象
- `JSON.stringify()`：将对象转化为字符串



## 日期

返回值为整数的API，数值为1970-1-1 00:00:00 UTC（世界标准时间）到某个时刻所经过的毫秒数：

- `Date.now()`：返回现在时刻。
- `Date.parse("2022-04-15T15:30:00.000+08:00")`：返回北京时间2022年4月15日 15:30:00的时刻。

与`Date`对象的**实例**相关的API：

- `new Date()`：返回现在时刻。
- `new Date("2022-04-15T15:30:00.000+08:00")`：返回北京时间2022年4月15日 15:30:00的时刻。
- 两个Date对象实例的差值为毫秒数
- `getDay()`：返回星期，0表示星期日，1-6表示星期一至星期六
- `getDate()`：返回日，数值为1-31
- `getMonth()`：返回月，数值为0-11
- `getFullYear()`：返回年份
- `getHours()`：返回小时
- `getMinutes()`：返回分钟
- `getSeconds()`：返回秒
- `getMilliseconds()`：返回毫秒

## WebSocket

与服务器建立全双工连接。

常用API：

- `new WebSocket('ws://localhost:8080');`：建立`ws`连接。
- `send()`：向服务器端发送一个字符串。一般用`JSON`将传入的对象序列化为字符串。
- `onopen`：类似于`onclick`，当连接建立时触发。
- `onmessage`：当从服务器端接收到消息时触发。
- `close()`：关闭连接。
- `onclose`：当连接关闭后触发。

## window

- `window.open("https://www.acwing.com")`在新标签栏中打开页面。
- `location.reload()`刷新页面。
- `location.href = "https://www.acwing.com"`：在当前标签栏中打开页面。

## canvas

https://developer.mozilla.org/zh-CN/docs/Web/API/Canvas_API/Tutorial

