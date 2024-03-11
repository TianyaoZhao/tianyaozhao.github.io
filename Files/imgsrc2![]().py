import re

def convert_img_tags(input_path, output_path):
    # 读取原始Markdown文件内容
    with open(input_path, 'r', encoding='utf-8') as f:
        markdown_content = f.read()

    # 使用正则表达式匹配所有<img>标签
    img_pattern = re.compile(r'<img\s+src="([^"]+)"[^>]*>', re.MULTILINE)
    matches = img_pattern.findall(markdown_content)

    # 替换所有<img>标签为新的Markdown格式
    for match in matches:
        original_tag = match
        new_tag = f'![]({original_tag})'
        markdown_content = re.sub(rf'<img\s+src="{re.escape(original_tag)}"[^>]*>', new_tag, markdown_content)

    # 将处理后的内容写入新的Markdown文件
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(markdown_content)


# 替换Markdown文件中的所有<img>标签
convert_img_tags('CPP_Algorithm.md', 'outa.md')
