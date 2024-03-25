# Latex

## latex基本概念

### latex源代码结构

```latex
\documentclass{...} % ... 为某文档类
% 导言区 使用宏包/进行文档的全局设置
\begin{document}
% 正文内容
\end{document}
% 此后内容会被忽略

```

### 宏包和文档类

#### 文档类

指定文档类

```latex
\documentclass[⟨options⟩]{⟨class-name⟩}
```

`<class-name>`

- article 文章格式的文档类，广泛用于科技论文、报告、说明文档等。
- report 长篇报告格式的文档类，具有章节结构，用于综述、长篇论文、简单
- 的书籍等。
- book 书籍文档类，包含章节结构和前言、正文、后记等结构。
- proc 基于article 文档类的一个简单的学术文档模板。
- slides 幻灯格式的文档类，使用无衬线字体。
- minimal 一个极其精简的文档类，只设定了纸张大小和基本字号，用作代码测
- 试的最小工作示例（Minimal Working Example）

`<options>`

全局地规定一些排版的参数，如字号、纸张大小、单双面

`例如`

调用article 文档类排版文章，指定纸张为A4 大小，基本字号为11pt，双面排版

```latex
\documentclass[11pt,twoside,a4paper]{article}
```

#### 宏包

调用宏包，可以一次性调用多个宏包，在⟨package-name⟩ 中用逗号隔开

```latex
\usepackage[⟨options⟩]{⟨package-name⟩}
```

### latex中用到的文件

.sty 宏包文件。宏包的名称与文件名一致

.cls 文档类文件。文档类名称与文件名一致。

.bib BIBTEX 参考文献数据库文件。

.bst BIBTEX 用到的参考文献格式模板。

### 文件组织形式

当导言区内容较多时，常常将其单独放置在一个.tex 文件中，再用\input 命令插入。复杂的图、表、代码等也会用类似的手段处理。

```latex
\input{<filename>}
```

