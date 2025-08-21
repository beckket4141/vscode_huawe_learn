# 字符与ASCII码相互转换示例

# 使用 ord() 函数将字符转换为ASCII码
def char_to_ascii(character):
    """将字符转换为ASCII码"""
    return ord(character)

# 使用 chr() 函数将ASCII码转换为字符
def ascii_to_char(ascii_code):
    """将ASCII码转换为字符"""
    return chr(ascii_code)

# 示例：字符转ASCII码
print("字符转ASCII码示例:")
char_example = 'A'
ascii_result = char_to_ascii(char_example)
print(f"字符 '{char_example}' 的ASCII码是: {ascii_result}")

char_example = 'a'
ascii_result = char_to_ascii(char_example)
print(f"字符 '{char_example}' 的ASCII码是: {ascii_result}")

char_example = '0'
ascii_result = char_to_ascii(char_example)
print(f"字符 '{char_example}' 的ASCII码是: {ascii_result}")

print("\n" + "="*30 + "\n")

# 示例：ASCII码转字符
print("ASCII码转字符示例:")
ascii_example = 65
char_result = ascii_to_char(ascii_example)
print(f"ASCII码 {ascii_example} 对应的字符是: '{char_result}'")

ascii_example = 97
char_result = ascii_to_char(ascii_example)
print(f"ASCII码 {ascii_example} 对应的字符是: '{char_result}'")

ascii_example = 48
char_result = ascii_to_char(ascii_example)
print(f"ASCII码 {ascii_example} 对应的字符是: '{char_result}'")

print("\n" + "="*30 + "\n")

# 更多示例：转换整个字符串
print("转换整个字符串示例:")
text = "Hello"
print(f"原始文本: {text}")

# 转换为ASCII码列表
ascii_list = [char_to_ascii(c) for c in text]
print(f"ASCII码列表: {ascii_list}")

# 从ASCII码列表转换回字符串
recovered_text = ''.join([ascii_to_char(code) for code in ascii_list])
print(f"恢复的文本: {recovered_text}")