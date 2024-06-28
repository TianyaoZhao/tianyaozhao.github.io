## tmux 教程

**功能**：

1. 分屏

2. 允许断开`Terminal`连接后，继续运行进程

**结构**：

一个`tmux`可以包含多个`session`，一个`session`可以包含多个`window`，一个`window`可以包含多个`pane`

```shell
tmux：
    session 0：
        window 0：
            pane 0
            pane 1
            pane 2
            ...
        window 1
        window 2
        ...
    session 1
    session 2
    ...
```

**操作**：

1. **`tmux`：新建一个`session`，其中包含一个`window`，`window`中包含一个`pane`，`pane`里打开了一个`shell`对话框**

2. **按下`Ctrl + a`后手指松开，然后按`%`：将当前`pane`左右平分成两个`pane`**

3. **按下`Ctrl + a`后手指松开，然后按`"`：将当前`pane`上下平分成两个`pane`**

4. **`Ctrl + d`：关闭当前`pane`;如果当前`window`的所有`pane`均已关闭，则自动关闭`window`如果当前`session`的所有`window`均已关闭，则自动关闭`session`**

5. 鼠标点击可以选择`pane`

6. 按下`Ctrl + a`后手指松开，然后按方向键：选择相邻的`pane`

7. 鼠标拖动`pane`之间的分割线，可以调整分割线的位置

8. 按下`Ctrl + a`的同时按方向键，可以调整pane之间分割线的位置

9. 按下`Ctrl + a`后手指松开，然后按`z`：将当前pane全屏/取消全屏

10. **按下`Ctrl + a`后手指松开，然后按`d`：挂起当前`session`**

11. **`tmux a`：打开之前挂起的`session`**

12. 按下`Ctrl + a`后手指松开，然后按s：**选择其它session**

    - 方向键 ——上：选择上一项 `session/window/pane`
    
    - 方向键 ——下：选择下一项 `session/window/pane`
    
    - 方向键 ——左：展开当前项 `session/window`
    
    - 方向键 ——右：闭合当前项 `session/window`


13. 按下`Ctrl + a`后手指松开，然后按`c`：在当前`session`中创建一个新的`window`

14. 按下`Ctrl + a`后手指松开，然后按`w`：选择其它`window`，操作方法与(12)一致

15. 按下`Ctrl + a`后手指松开，然后按`Page Up`：翻阅当前`pane`内的内容

16. 鼠标滚轮：翻阅当前`pane`内的内容

17. **在`tmux`中选中文本时，需要按住`Shift`键**

18. **`tmux`中复制/粘贴文本的通用方式：**

    - **复制：`ctrl + insert`**


    - **粘贴：`shift + insert`**



